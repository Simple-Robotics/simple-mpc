#include "simple-mpc/centroidal-dynamics.hpp"

#include <aligator/modelling/centroidal/angular-acceleration.hpp>
#include <aligator/modelling/centroidal/angular-momentum.hpp>
#include <aligator/modelling/centroidal/centroidal-acceleration.hpp>
#include <aligator/modelling/centroidal/centroidal-friction-cone.hpp>
#include <aligator/modelling/centroidal/centroidal-translation.hxx>
#include <aligator/modelling/centroidal/centroidal-wrench-cone.hpp>
#include <aligator/modelling/centroidal/linear-momentum.hpp>
#include <aligator/modelling/dynamics/centroidal-fwd.hpp>
#include <aligator/modelling/dynamics/integrator-euler.hpp>

namespace simple_mpc
{
  using namespace aligator;
  using VectorSpace = aligator::VectorSpaceTpl<double>;
  using CentroidalFwdDynamics = dynamics::CentroidalFwdDynamicsTpl<double>;
  using CentroidalAccelerationResidual = CentroidalAccelerationResidualTpl<double>;
  using AngularAccelerationResidual = AngularAccelerationResidualTpl<double>;
  using LinearMomentumResidual = LinearMomentumResidualTpl<double>;
  using AngularMomentumResidual = AngularMomentumResidualTpl<double>;
  using CentroidalCoMResidual = CentroidalCoMResidualTpl<double>;
  using CentroidalWrenchConeResidual = CentroidalWrenchConeResidualTpl<double>;
  using CentroidalFrictionConeResidual = CentroidalFrictionConeResidualTpl<double>;
  using IntegratorEuler = dynamics::IntegratorEulerTpl<double>;

  CentroidalOCP::CentroidalOCP(const CentroidalSettings & settings, const RobotModelHandler & model_handler)
  : Base(model_handler)
  , settings_(settings)
  {
    nx_ = 9;
    nu_ = (int)model_handler_.getFeetNb() * settings_.force_size;
    control_ref_.resize(nu_);
    control_ref_.setZero();
    com_ref_.setZero();
    x0_.resize(9);
  }

  StageModel CentroidalOCP::createStage(
    const std::map<std::string, bool> & contact_phase,
    const std::map<std::string, pinocchio::SE3> & contact_pose,
    const std::map<std::string, Eigen::VectorXd> & contact_force,
    const std::map<std::string, bool> & /*land_constraint*/)
  {
    auto space = VectorSpace(nx_);
    auto rcost = CostStack(space, nu_);
    std::vector<bool> contact_states;
    StdVectorEigenAligned<Eigen::Vector3d> contact_poses;

    for (auto const & x : contact_phase)
    {
      contact_states.push_back(x.second);
    }
    for (auto const & x : contact_pose)
    {
      contact_poses.push_back(x.second.translation());
    }

    computeControlFromForces(contact_force);

    ContactMap contact_map = ContactMap(model_handler_.getFeetFrameNames(), contact_states, contact_poses);

    auto com_res = CentroidalCoMResidual(nx_, nu_, com_ref_);
    auto linear_mom = LinearMomentumResidual(nx_, nu_, Eigen::Vector3d::Zero());
    auto angular_mom = AngularMomentumResidual(nx_, nu_, Eigen::Vector3d::Zero());

    auto linear_acc = CentroidalAccelerationResidual(
      space.ndx(), nu_, model_handler_.getMass(), settings_.gravity, contact_map, settings_.force_size);
    auto angular_acc = AngularAccelerationResidual(
      space.ndx(), nu_, model_handler_.getMass(), settings_.gravity, contact_map, settings_.force_size);

    rcost.addCost("com_cost", QuadraticResidualCost(space, com_res, settings_.w_com));
    rcost.addCost("control_cost", QuadraticControlCost(space, control_ref_, settings_.w_u));
    rcost.addCost("linear_mom_cost", QuadraticResidualCost(space, linear_mom, settings_.w_linear_mom));
    rcost.addCost("angular_mom_cost", QuadraticResidualCost(space, angular_mom, settings_.w_angular_mom));
    rcost.addCost("linear_acc_cost", QuadraticResidualCost(space, linear_acc, settings_.w_linear_acc));
    rcost.addCost("angular_acc_cost", QuadraticResidualCost(space, angular_acc, settings_.w_angular_acc));

    CentroidalFwdDynamics ode =
      CentroidalFwdDynamics(space, model_handler_.getMass(), settings_.gravity, contact_map, settings_.force_size);
    IntegratorEuler dyn_model = IntegratorEuler(ode, settings_.timestep);

    StageModel stm = StageModel(rcost, dyn_model);

    int i = 0;
    for (auto const & name : model_handler_.getFeetFrameNames())
    {
      if (contact_phase.at(name))
      {
        if (settings_.force_size == 6)
        {
          CentroidalWrenchConeResidual wrench_residual =
            CentroidalWrenchConeResidual(space.ndx(), nu_, i, settings_.mu, settings_.Lfoot, settings_.Wfoot);
          stm.addConstraint(wrench_residual, NegativeOrthant());
        }
        else
        {
          CentroidalFrictionConeResidual friction_residual =
            CentroidalFrictionConeResidual(space.ndx(), nu_, i, settings_.mu, 1e-4);
          stm.addConstraint(friction_residual, NegativeOrthant());
        }
      }
      i++;
    }
    return stm;
  }

  void CentroidalOCP::computeControlFromForces(const std::map<std::string, Eigen::VectorXd> & force_refs)
  {
    for (std::size_t i = 0; i < model_handler_.getFeetNb(); i++)
    {
      if (settings_.force_size != force_refs.at(model_handler_.getFootFrameName(i)).size())
      {
        throw std::runtime_error("force size in settings does not match reference force size");
      }
      control_ref_.segment((long)i * settings_.force_size, settings_.force_size) =
        force_refs.at(model_handler_.getFootFrameName(i));
    }
  }

  void CentroidalOCP::setReferencePoses(const std::size_t t, const std::map<std::string, pinocchio::SE3> & pose_refs)
  {
    if (t >= problem_->stages_.size())
    {
      throw std::runtime_error("Stage index exceeds stage vector size");
    }
    if (pose_refs.size() != model_handler_.getFeetNb())
    {
      throw std::runtime_error("pose_refs size does not match number of end effectors");
    }
    CentroidalFwdDynamics * cent_dyn =
      problem_->stages_[t]->getDynamics<IntegratorEuler>()->getDynamics<CentroidalFwdDynamics>();

    for (auto const & pose : pose_refs)
    {
      cent_dyn->contact_map_.setContactPose(pose.first, pose.second.translation());
    }
    CostStack * cs = getCostStack(t);

    for (auto ee_name : model_handler_.getFeetFrameNames())
    {
      QuadraticResidualCost * qrc1 = cs->getComponent<QuadraticResidualCost>("linear_acc_cost");
      QuadraticResidualCost * qrc2 = cs->getComponent<QuadraticResidualCost>("angular_acc_cost");
      CentroidalAccelerationResidual * car = qrc1->getResidual<CentroidalAccelerationResidual>();
      AngularAccelerationResidual * aar = qrc2->getResidual<AngularAccelerationResidual>();
      car->contact_map_.setContactPose(ee_name, pose_refs.at(ee_name).translation());
      aar->contact_map_.setContactPose(ee_name, pose_refs.at(ee_name).translation());
    }
  }

  void
  CentroidalOCP::setReferencePose(const std::size_t t, const std::string & ee_name, const pinocchio::SE3 & pose_ref)
  {
    if (t >= problem_->stages_.size())
    {
      throw std::runtime_error("Stage index exceeds stage vector size");
    }
    CentroidalFwdDynamics * cent_dyn =
      problem_->stages_[t]->getDynamics<IntegratorEuler>()->getDynamics<CentroidalFwdDynamics>();
    cent_dyn->contact_map_.setContactPose(ee_name, pose_ref.translation());

    CostStack * cs = getCostStack(t);
    QuadraticResidualCost * qrc1 = cs->getComponent<QuadraticResidualCost>("linear_acc_cost");
    QuadraticResidualCost * qrc2 = cs->getComponent<QuadraticResidualCost>("angular_acc_cost");
    CentroidalAccelerationResidual * car = qrc1->getResidual<CentroidalAccelerationResidual>();
    AngularAccelerationResidual * aar = qrc2->getResidual<AngularAccelerationResidual>();
    car->contact_map_.setContactPose(ee_name, pose_ref.translation());
    aar->contact_map_.setContactPose(ee_name, pose_ref.translation());
  }

  const pinocchio::SE3 CentroidalOCP::getReferencePose(const std::size_t t, const std::string & ee_name)
  {
    if (t >= problem_->stages_.size())
    {
      throw std::runtime_error("Stage index exceeds stage vector size");
    }
    CentroidalFwdDynamics * cent_dyn =
      problem_->stages_[t]->getDynamics<IntegratorEuler>()->getDynamics<CentroidalFwdDynamics>();

    pinocchio::SE3 pose = pinocchio::SE3::Identity();
    pose.translation() = cent_dyn->contact_map_.getContactPose(ee_name);

    return pose;
  }

  void CentroidalOCP::setReferenceForces(const std::size_t t, const std::map<std::string, Eigen::VectorXd> & force_refs)
  {
    computeControlFromForces(force_refs);
    setReferenceControl(t, control_ref_);
  }

  void
  CentroidalOCP::setReferenceForce(const std::size_t t, const std::string & ee_name, const ConstVectorRef & force_ref)
  {
    std::vector<std::string> hname = model_handler_.getFeetFrameNames();
    std::vector<std::string>::iterator it = std::find(hname.begin(), hname.end(), ee_name);
    long id = it - hname.begin();
    control_ref_.segment(id * settings_.force_size, settings_.force_size) = force_ref;
    setReferenceControl(t, control_ref_);
  }

  const Eigen::VectorXd CentroidalOCP::getReferenceForce(const std::size_t t, const std::string & ee_name)
  {
    std::vector<std::string> hname = model_handler_.getFeetFrameNames();
    std::vector<std::string>::iterator it = std::find(hname.begin(), hname.end(), ee_name);
    long id = it - hname.begin();

    return getReferenceControl(t).segment(id * settings_.force_size, settings_.force_size);
  }

  const Eigen::VectorXd CentroidalOCP::getVelocityBase(const std::size_t t)
  {
    CostStack * cs = getCostStack(t);
    Eigen::VectorXd v(6);

    QuadraticResidualCost * qcm = cs->getComponent<QuadraticResidualCost>("linear_mom_cost");
    LinearMomentumResidual * cfm = qcm->getResidual<LinearMomentumResidual>();

    QuadraticResidualCost * qca = cs->getComponent<QuadraticResidualCost>("angular_mom_cost");
    AngularMomentumResidual * cfa = qca->getResidual<AngularMomentumResidual>();

    v.head<3>() = cfm->getReference() / model_handler_.getMass();
    v.tail<3>() = cfa->getReference() / model_handler_.getMass();
    return v;
  }

  void CentroidalOCP::setVelocityBase(const std::size_t t, const ConstVectorRef & velocity_base)
  {
    assert(velocity_base.size() == 6 && "velocity_base not of the right size");
    CostStack * cs = getCostStack(t);
    QuadraticResidualCost * qcm = cs->getComponent<QuadraticResidualCost>("linear_mom_cost");
    LinearMomentumResidual * cfm = qcm->getResidual<LinearMomentumResidual>();

    QuadraticResidualCost * qca = cs->getComponent<QuadraticResidualCost>("angular_mom_cost");
    AngularMomentumResidual * cfa = qca->getResidual<AngularMomentumResidual>();

    cfm->setReference(velocity_base.head<3>() * model_handler_.getMass());
    cfa->setReference(velocity_base.tail<3>() * model_handler_.getMass());
  }

  const Eigen::VectorXd CentroidalOCP::getPoseBase(const std::size_t t)
  {
    CostStack * cs = getCostStack(t);
    QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>("com_cost");
    CentroidalCoMResidual * cfr = qrc->getResidual<CentroidalCoMResidual>();
    return cfr->getReference();
  }

  void CentroidalOCP::setPoseBase(const std::size_t t, const ConstVectorRef & pose_base)
  {
    assert(pose_base.size() == 3 && "pose_base not of the right size");
    CostStack * cs = getCostStack(t);
    QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>("com_cost");
    CentroidalCoMResidual * cfr = qrc->getResidual<CentroidalCoMResidual>();
    com_ref_ = pose_base.head<3>();
    cfr->setReference(com_ref_);
  }

  const Eigen::VectorXd CentroidalOCP::getProblemState(const RobotDataHandler & data_handler)
  {
    return data_handler.getCentroidalState();
  }

  size_t CentroidalOCP::getContactSupport(const std::size_t t)
  {
    CentroidalFwdDynamics * ode =
      problem_->stages_[t]->getDynamics<IntegratorEuler>()->getDynamics<CentroidalFwdDynamics>();

    size_t active_contacts = 0;
    for (auto name : model_handler_.getFeetFrameNames())
    {
      if (ode->contact_map_.getContactState(name))
      {
        active_contacts += 1;
      }
    }
    return active_contacts;
  }

  std::vector<bool> CentroidalOCP::getContactState(const std::size_t t)
  {
    std::vector<bool> contact_state;
    CentroidalFwdDynamics * ode =
      problem_->stages_[t]->getDynamics<IntegratorEuler>()->getDynamics<CentroidalFwdDynamics>();
    assert(ode != nullptr);
    for (auto name : model_handler_.getFeetFrameNames())
      contact_state.push_back(ode->contact_map_.getContactState(name));

    return contact_state;
  }

  void CentroidalOCP::setReferenceState(const std::size_t t, const ConstVectorRef & x_ref)
  {
    assert(x_ref.size() == 9 && "x_ref not of the right size");
    setPoseBase(t, x_ref.head<3>());
    setVelocityBase(t, x_ref.tail<6>());
  }

  const ConstVectorRef CentroidalOCP::getReferenceState(const std::size_t t)
  {
    x0_.head<3>() = getPoseBase(t);
    x0_.tail<6>() = getVelocityBase(t);
    return x0_;
  }

  CostStack CentroidalOCP::createTerminalCost()
  {
    auto ter_space = VectorSpace(nx_);
    auto term_cost = CostStack(ter_space, nu_);
    auto linear_mom = LinearMomentumResidual(nx_, nu_, Eigen::Vector3d::Zero());
    auto angular_mom = AngularMomentumResidual(nx_, nu_, Eigen::Vector3d::Zero());
    term_cost.addCost("linear_mom_cost", QuadraticResidualCost(ter_space, linear_mom, settings_.w_linear_mom));
    term_cost.addCost("angular_mom_cost", QuadraticResidualCost(ter_space, angular_mom, settings_.w_angular_mom));

    return term_cost;
  }

  void CentroidalOCP::createTerminalConstraint(const Eigen::Vector3d & com_ref)
  {
    if (!problem_initialized_)
    {
      throw std::runtime_error("Create problem first!");
    }
    CentroidalCoMResidual com_cstr = CentroidalCoMResidual(ndx_, nu_, com_ref);

    // problem_->addTerminalConstraint(com_cstr, EqualityConstraint());
    terminal_constraint_ = false;
  }

  void CentroidalOCP::updateTerminalConstraint(const Eigen::Vector3d & com_ref)
  {
    if (terminal_constraint_)
    {
      CentroidalCoMResidual * CoMres = problem_->term_cstrs_.getConstraint<CentroidalCoMResidual>(0);

      CoMres->setReference(com_ref);
    }
  }

} // namespace simple_mpc
