#include "simple-mpc/kinodynamics.hpp"

#include <aligator/modelling/centroidal/centroidal-friction-cone.hpp>
#include <aligator/modelling/centroidal/centroidal-wrench-cone.hpp>
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <aligator/modelling/dynamics/kinodynamics-fwd.hpp>
#include <aligator/modelling/multibody/centroidal-momentum-derivative.hpp>
#include <aligator/modelling/multibody/centroidal-momentum.hpp>
#include <aligator/modelling/multibody/frame-placement.hpp>
#include <aligator/modelling/multibody/frame-translation.hpp>
#include <aligator/modelling/multibody/frame-velocity.hpp>
#include <aligator/modelling/multibody/dcm-position.hpp>

namespace simple_mpc
{
  using namespace aligator;
  using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;
  using KinodynamicsFwdDynamics = dynamics::KinodynamicsFwdDynamicsTpl<double>;
  using CentroidalMomentumDerivativeResidual = CentroidalMomentumDerivativeResidualTpl<double>;
  using CentroidalMomentumResidual = CentroidalMomentumResidualTpl<double>;
  using CentroidalWrenchConeResidual = CentroidalWrenchConeResidualTpl<double>;
  using CentroidalFrictionConeResidual = CentroidalFrictionConeResidualTpl<double>;
  using FramePlacementResidual = FramePlacementResidualTpl<double>;
  using FrameTranslationResidual = FrameTranslationResidualTpl<double>;
  using FrameVelocityResidual = FrameVelocityResidualTpl<double>;
  using IntegratorSemiImplEuler = dynamics::IntegratorSemiImplEulerTpl<double>;
  using DCMPositionResidual = DCMPositionResidualTpl<double>;

  KinodynamicsOCP::KinodynamicsOCP(const KinodynamicsSettings & settings, const RobotModelHandler & model_handler)
  : Base(model_handler)
  , settings_(settings)
  {

    nu_ = nv_ - 6 + settings_.force_size * (int)model_handler_.getFeetNames().size();
    x0_ = model_handler_.getReferenceState();
    control_ref_.resize(nu_);
    control_ref_.setZero();
  }

  StageModel KinodynamicsOCP::createStage(
    const std::map<std::string, bool> & contact_phase,
    const std::map<std::string, pinocchio::SE3> & contact_pose,
    const std::map<std::string, Eigen::VectorXd> & contact_force,
    const std::map<std::string, bool> & land_constraint)
  {
    auto space = MultibodyPhaseSpace(model_handler_.getModel());
    auto rcost = CostStack(space, nu_);
    std::vector<bool> contact_states;
    for (auto const & x : contact_phase)
    {
      contact_states.push_back(x.second);
    }

    computeControlFromForces(contact_force);

    auto cent_mom = CentroidalMomentumResidual(space.ndx(), nu_, model_handler_.getModel(), Eigen::VectorXd::Zero(6));
    auto centder_mom = CentroidalMomentumDerivativeResidual(
      space.ndx(), model_handler_.getModel(), settings_.gravity, contact_states, model_handler_.getFeetIds(),
      settings_.force_size);
    rcost.addCost("state_cost", QuadraticStateCost(space, nu_, model_handler_.getReferenceState(), settings_.w_x));
    rcost.addCost("control_cost", QuadraticControlCost(space, control_ref_, settings_.w_u));
    rcost.addCost("centroidal_cost", QuadraticResidualCost(space, cent_mom, settings_.w_cent));
    rcost.addCost("centroidal_derivative_cost", QuadraticResidualCost(space, centder_mom, settings_.w_centder));

    for (auto const & name : model_handler_.getFeetNames())
    {
      if (settings_.force_size == 6)
      {
        FramePlacementResidual frame_residual = FramePlacementResidual(
          space.ndx(), nu_, model_handler_.getModel(), contact_pose.at(name), model_handler_.getFootId(name));

        rcost.addCost(name + "_pose_cost", QuadraticResidualCost(space, frame_residual, settings_.w_frame));
      }
      else
      {
        FrameTranslationResidual frame_residual = FrameTranslationResidual(
          space.ndx(), nu_, model_handler_.getModel(), contact_pose.at(name).translation(),
          model_handler_.getFootId(name));

        rcost.addCost(name + "_pose_cost", QuadraticResidualCost(space, frame_residual, settings_.w_frame));
      }
    }

    KinodynamicsFwdDynamics ode = KinodynamicsFwdDynamics(
      space, model_handler_.getModel(), settings_.gravity, contact_states, model_handler_.getFeetIds(),
      settings_.force_size);
    IntegratorSemiImplEuler dyn_model = IntegratorSemiImplEuler(ode, settings_.timestep);
    StageModel stm = StageModel(rcost, dyn_model);

    if (settings_.kinematics_limits)
    {
      StateErrorResidual state_fn = StateErrorResidual(space, nu_, space.neutral());
      std::vector<int> state_id;
      for (int i = 6; i < nv_; i++)
      {
        state_id.push_back(i);
      }
      FunctionSliceXpr state_slice = FunctionSliceXpr(state_fn, state_id);
      stm.addConstraint(state_slice, BoxConstraint(settings_.qmin, settings_.qmax));
    }

    Motion v_ref = Motion::Zero();
    int i = 0;
    for (auto const & name : model_handler_.getFeetNames())
    {
      if (contact_phase.at(name))
      {
        FrameVelocityResidual frame_vel = FrameVelocityResidual(
          space.ndx(), nu_, model_handler_.getModel(), v_ref, model_handler_.getFootId(name), pinocchio::LOCAL);
        if (settings_.force_size == 6)
        {
          if (settings_.force_cone)
          {
            CentroidalWrenchConeResidual wrench_residual =
              CentroidalWrenchConeResidual(space.ndx(), nu_, i, settings_.mu, settings_.Lfoot, settings_.Wfoot);
            stm.addConstraint(wrench_residual, NegativeOrthant());
          }
          stm.addConstraint(frame_vel, EqualityConstraint());
        }
        else
        {
          if (settings_.force_cone)
          {
            CentroidalFrictionConeResidual friction_residual =
              CentroidalFrictionConeResidual(space.ndx(), nu_, i, settings_.mu, 1e-4);
            stm.addConstraint(friction_residual, NegativeOrthant());
          }
          std::vector<int> vel_id = {0, 1, 2};

          FunctionSliceXpr vel_slice = FunctionSliceXpr(frame_vel, vel_id);
          stm.addConstraint(vel_slice, EqualityConstraint());
          if (settings_.land_cstr and land_constraint.at(name))
          {
            std::vector<int> frame_id = {2};

            FrameTranslationResidual frame_residual = FrameTranslationResidual(
              space.ndx(), nu_, model_handler_.getModel(), contact_pose.at(name).translation(),
              model_handler_.getFootId(name));

            FunctionSliceXpr frame_slice = FunctionSliceXpr(frame_residual, frame_id);

            stm.addConstraint(frame_slice, EqualityConstraint());
          }
        }
      }
      i++;
    }

    return stm;
  }

  void
  KinodynamicsOCP::setReferencePose(const std::size_t t, const std::string & ee_name, const pinocchio::SE3 & pose_ref)
  {
    CostStack * cs = getCostStack(t);
    QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>(ee_name + "_pose_cost");
    if (settings_.force_size == 6)
    {
      FramePlacementResidual * cfr = qrc->getResidual<FramePlacementResidual>();
      cfr->setReference(pose_ref);
    }
    else
    {
      FrameTranslationResidual * cfr = qrc->getResidual<FrameTranslationResidual>();
      cfr->setReference(pose_ref.translation());
    }
  }

  void KinodynamicsOCP::setReferencePoses(const std::size_t t, const std::map<std::string, pinocchio::SE3> & pose_refs)
  {
    if (pose_refs.size() != model_handler_.getFeetNames().size())
    {
      throw std::runtime_error("pose_refs size does not match number of end effectors");
    }

    CostStack * cs = getCostStack(t);
    for (auto ee_name : model_handler_.getFeetNames())
    {
      QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>(ee_name + "_pose_cost");
      if (settings_.force_size == 6)
      {
        FramePlacementResidual * cfr = qrc->getResidual<FramePlacementResidual>();
        cfr->setReference(pose_refs.at(ee_name));
      }
      else
      {
        FrameTranslationResidual * cfr = qrc->getResidual<FrameTranslationResidual>();
        cfr->setReference(pose_refs.at(ee_name).translation());
      }
    }
  }

  void KinodynamicsOCP::setTerminalReferencePose(const std::string & ee_name, const pinocchio::SE3 & pose_ref)
  {
    CostStack * cs = getTerminalCostStack();
    QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>(ee_name + "_pose_cost");
    if (settings_.force_size == 6)
    {
      FramePlacementResidual * cfr = qrc->getResidual<FramePlacementResidual>();
      cfr->setReference(pose_ref);
    }
    else
    {
      FrameTranslationResidual * cfr = qrc->getResidual<FrameTranslationResidual>();
      cfr->setReference(pose_ref.translation());
    }
  }

  const pinocchio::SE3 KinodynamicsOCP::getReferencePose(const std::size_t t, const std::string & ee_name)
  {
    CostStack * cs = getCostStack(t);
    QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>(ee_name + "_pose_cost");
    if (settings_.force_size == 6)
    {
      FramePlacementResidual * cfr = qrc->getResidual<FramePlacementResidual>();
      return cfr->getReference();
    }
    else
    {
      FrameTranslationResidual * cfr = qrc->getResidual<FrameTranslationResidual>();
      SE3 ref = SE3::Identity();
      ref.translation() = cfr->getReference();
      return ref;
    }
  }

  void KinodynamicsOCP::computeControlFromForces(const std::map<std::string, Eigen::VectorXd> & force_refs)
  {
    for (std::size_t i = 0; i < model_handler_.getFeetNames().size(); i++)
    {
      if (settings_.force_size != force_refs.at(model_handler_.getFootName(i)).size())
      {
        throw std::runtime_error("force size in settings does not match reference force size");
      }
      control_ref_.segment((long)i * settings_.force_size, settings_.force_size) =
        force_refs.at(model_handler_.getFootName(i));
    }
  }

  void
  KinodynamicsOCP::setReferenceForces(const std::size_t i, const std::map<std::string, Eigen::VectorXd> & force_refs)
  {
    computeControlFromForces(force_refs);
    setReferenceControl(i, control_ref_);
  }

  void
  KinodynamicsOCP::setReferenceForce(const std::size_t i, const std::string & ee_name, const ConstVectorRef & force_ref)
  {
    std::vector<std::string> hname = model_handler_.getFeetNames();
    std::vector<std::string>::iterator it = std::find(hname.begin(), hname.end(), ee_name);
    long id = it - hname.begin();
    control_ref_.segment(id * settings_.force_size, settings_.force_size) = force_ref;
    setReferenceControl(i, control_ref_);
  }

  const Eigen::VectorXd KinodynamicsOCP::getReferenceForce(const std::size_t i, const std::string & ee_name)
  {
    std::vector<std::string> hname = model_handler_.getFeetNames();
    std::vector<std::string>::iterator it = std::find(hname.begin(), hname.end(), ee_name);
    long id = it - hname.begin();

    return getReferenceControl(i).segment(id * settings_.force_size, settings_.force_size);
  }

  const Eigen::VectorXd KinodynamicsOCP::getVelocityBase(const std::size_t t)
  {
    CostStack * cs = getCostStack(t);
    QuadraticStateCost * qc = cs->getComponent<QuadraticStateCost>("state_cost");
    return qc->getTarget().segment(nq_, 6);
  }

  void KinodynamicsOCP::setVelocityBase(const std::size_t t, const ConstVectorRef & velocity_base)
  {
    if (velocity_base.size() != 6)
    {
      throw std::runtime_error("velocity_base size should be 6");
    }
    CostStack * cs = getCostStack(t);
    QuadraticStateCost * qc = cs->getComponent<QuadraticStateCost>("state_cost");
    x0_ = getReferenceState(t);
    x0_.segment(nq_, 6) = velocity_base;
    qc->setTarget(x0_);
  }

  const Eigen::VectorXd KinodynamicsOCP::getPoseBase(const std::size_t t)
  {
    CostStack * cs = getCostStack(t);
    QuadraticStateCost * qc = cs->getComponent<QuadraticStateCost>("state_cost");
    return qc->getTarget().head(7);
  };

  void KinodynamicsOCP::setPoseBase(const std::size_t t, const ConstVectorRef & pose_base)
  {
    if (pose_base.size() != 7)
    {
      throw std::runtime_error("pose_base size should be 7");
    }
    CostStack * cs = getCostStack(t);
    QuadraticStateCost * qc = cs->getComponent<QuadraticStateCost>("state_cost");
    x0_ = getReferenceState(t);
    x0_.head(7) = pose_base;
    qc->setTarget(x0_);
  }

  const Eigen::VectorXd KinodynamicsOCP::getProblemState(const RobotDataHandler & data_handler)
  {
    return data_handler.getState();
  }

  void KinodynamicsOCP::setReferenceState(const std::size_t t, const ConstVectorRef & x_ref)
  {
    assert(x_ref.size() == nq_ + nv_ && "x_ref not of the right size");
    CostStack * cs = getCostStack(t);
    QuadraticStateCost * qc = cs->getComponent<QuadraticStateCost>("state_cost");
    qc->setTarget(x_ref);
  }

  const ConstVectorRef KinodynamicsOCP::getReferenceState(const std::size_t t)
  {
    CostStack * cs = getCostStack(t);
    QuadraticStateCost * qc = cs->getComponent<QuadraticStateCost>("state_cost");
    return qc->getTarget();
  }

  size_t KinodynamicsOCP::getContactSupport(const std::size_t t)
  {
    KinodynamicsFwdDynamics * ode =
      problem_->stages_[t]->getDynamics<IntegratorSemiImplEuler>()->getDynamics<KinodynamicsFwdDynamics>();

    size_t active_contacts = 0;
    for (auto const contact : ode->contact_states_)
    {
      if (contact)
      {
        active_contacts += 1;
      }
    }
    return active_contacts;
  }

  std::vector<bool> KinodynamicsOCP::getContactState(const std::size_t t)
  {
    KinodynamicsFwdDynamics * ode =
      problem_->stages_[t]->getDynamics<IntegratorSemiImplEuler>()->getDynamics<KinodynamicsFwdDynamics>();
    assert(ode != nullptr);
    return ode->contact_states_;
  }

  CostStack KinodynamicsOCP::createTerminalCost()
  {
    auto ter_space = MultibodyPhaseSpace(model_handler_.getModel());
    auto term_cost = CostStack(ter_space, nu_);
    auto cent_mom =
      CentroidalMomentumResidual(ter_space.ndx(), nu_, model_handler_.getModel(), Eigen::VectorXd::Zero(6));

    term_cost.addCost(
      "state_cost", QuadraticStateCost(ter_space, nu_, model_handler_.getReferenceState(), settings_.w_x));
    term_cost.addCost("centroidal_cost", QuadraticResidualCost(ter_space, cent_mom, settings_.w_cent * 10));

    return term_cost;
  }

  void KinodynamicsOCP::createTerminalConstraint(const Eigen::Vector3d & com_ref)
  {
    if (!problem_initialized_)
    {
      throw std::runtime_error("Create problem first!");
    }
    double tau = sqrt(com_ref[2] / 9.81);
    DCMPositionResidual dcm_cstr = DCMPositionResidual(ndx_, nu_, model_handler_.getModel(), com_ref, tau);

    problem_->addTerminalConstraint(dcm_cstr, EqualityConstraint());
    terminal_constraint_ = true;
  }

  void KinodynamicsOCP::updateTerminalConstraint(const Eigen::Vector3d & com_ref)
  {
    if (terminal_constraint_)
    {
      DCMPositionResidual * DCMres =
        problem_->term_cstrs_.getConstraint<DCMPositionResidual>(0);

      DCMres->setReference(com_ref);
    }
  }

} // namespace simple_mpc
