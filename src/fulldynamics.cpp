#include "simple-mpc/fulldynamics.hpp"
#include "simple-mpc/ocp-handler.hpp"

#include "aligator/modelling/dynamics/multibody-constraint-fwd.hpp"
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <aligator/modelling/multibody/center-of-mass-translation.hpp>
#include <aligator/modelling/multibody/centroidal-momentum.hpp>
#include <aligator/modelling/multibody/contact-force.hpp>
#include <aligator/modelling/multibody/dcm-position.hpp>
#include <aligator/modelling/multibody/frame-placement.hpp>
#include <aligator/modelling/multibody/frame-translation.hpp>
#include <aligator/modelling/multibody/frame-velocity.hpp>
#include <aligator/modelling/multibody/multibody-friction-cone.hpp>
#include <aligator/modelling/multibody/multibody-wrench-cone.hpp>

namespace simple_mpc
{
  using namespace aligator;
  using ContactForceResidual = ContactForceResidualTpl<double>;
  using CentroidalMomentumResidual = CentroidalMomentumResidualTpl<double>;
  using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;
  using MultibodyWrenchConeResidual = aligator::MultibodyWrenchConeResidualTpl<double>;
  using MultibodyFrictionConeResidual = MultibodyFrictionConeResidualTpl<double>;
  using MultibodyConstraintFwdDynamics = dynamics::MultibodyConstraintFwdDynamicsTpl<double>;
  using FramePlacementResidual = FramePlacementResidualTpl<double>;
  using FrameTranslationResidual = FrameTranslationResidualTpl<double>;
  using FrameVelocityResidual = FrameVelocityResidualTpl<double>;
  using DCMPositionResidual = DCMPositionResidualTpl<double>;
  using CenterOfMassTranslationResidual = CenterOfMassTranslationResidualTpl<double>;
  using IntegratorSemiImplEuler = dynamics::IntegratorSemiImplEulerTpl<double>;

  FullDynamicsOCP::FullDynamicsOCP(const FullDynamicsSettings & settings, const RobotModelHandler & model_handler)
  : Base(model_handler)
  , settings_(settings)
  {

    actuation_matrix_.resize(nv_, nu_);
    actuation_matrix_.setZero();
    actuation_matrix_.bottomRows(nu_).setIdentity();

    prox_settings_ = ProximalSettings(1e-9, 1e-10, 10);
    x0_ = model_handler_.getReferenceState();
    if (settings.force_size != settings.Kp_correction.size())
    {
      throw std::runtime_error("Force must be of same size as Kp correction");
    }
    if (settings.force_size != settings.Kd_correction.size())
    {
      throw std::runtime_error("Force must be of same size as Kd correction");
    }

    for (auto const & name : model_handler_.getFeetNames())
    {
      auto frame_ids = model_handler_.getFootId(name);
      auto joint_ids = model_handler_.getModel().frames[frame_ids].parentJoint;
      pinocchio::SE3 pl1 = model_handler_.getModel().frames[frame_ids].placement;
      pinocchio::SE3 pl2 = pinocchio::SE3::Identity();
      if (settings_.force_size == 6)
      {
        pinocchio::RigidConstraintModel constraint_model = pinocchio::RigidConstraintModel(
          pinocchio::ContactType::CONTACT_6D, model_handler_.getModel(), joint_ids, pl1, 0, pl2,
          pinocchio::LOCAL_WORLD_ALIGNED);
        constraint_model.corrector.Kp = settings.Kp_correction;
        constraint_model.corrector.Kd = settings.Kd_correction;
        constraint_model.name = name;
        constraint_models_.push_back(constraint_model);
      }
      else
      {
        pinocchio::RigidConstraintModel constraint_model = pinocchio::RigidConstraintModel(
          pinocchio::ContactType::CONTACT_3D, model_handler_.getModel(), joint_ids, pl1, 0, pl2,
          pinocchio::LOCAL_WORLD_ALIGNED);
        constraint_model.corrector.Kp = settings.Kp_correction;
        constraint_model.corrector.Kd = settings.Kd_correction;
        constraint_model.name = name;
        constraint_models_.push_back(constraint_model);
      }
    }
  }

  StageModel FullDynamicsOCP::createStage(
    const std::map<std::string, bool> & contact_phase,
    const std::map<std::string, pinocchio::SE3> & contact_pose,
    const std::map<std::string, Eigen::VectorXd> & contact_force,
    const std::map<std::string, bool> & land_constraint)
  {

    auto space = MultibodyPhaseSpace(model_handler_.getModel());
    auto rcost = CostStack(space, nu_);

    rcost.addCost("state_cost", QuadraticStateCost(space, nu_, x0_, settings_.w_x));
    rcost.addCost("control_cost", QuadraticControlCost(space, Eigen::VectorXd::Zero(nu_), settings_.w_u));

    auto cent_mom = CentroidalMomentumResidual(space.ndx(), nu_, model_handler_.getModel(), Eigen::VectorXd::Zero(6));
    rcost.addCost("centroidal_cost", QuadraticResidualCost(space, cent_mom, settings_.w_cent));

    pinocchio::context::RigidConstraintModelVector cms;

    size_t c_id = 0;
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

      if (contact_phase.at(name))
        cms.push_back(constraint_models_[c_id]);

      c_id++;
    }

    for (auto const & name : model_handler_.getFeetNames())
    {
      std::shared_ptr<ContactForceResidual> frame_force;
      if (contact_force.at(name).size() != settings_.force_size)
      {
        throw std::runtime_error("Reference forces do not have the right dimension");
      }
      if (contact_phase.at(name))
      {
        frame_force = std::make_shared<ContactForceResidual>(
          space.ndx(), model_handler_.getModel(), actuation_matrix_, cms, prox_settings_, contact_force.at(name), name);

        rcost.addCost(name + "_force_cost", QuadraticResidualCost(space, *frame_force, settings_.w_forces));
      }
    }

    MultibodyConstraintFwdDynamics ode = MultibodyConstraintFwdDynamics(space, actuation_matrix_, cms, prox_settings_);
    IntegratorSemiImplEuler dyn_model = IntegratorSemiImplEuler(ode, settings_.timestep);

    StageModel stm = StageModel(rcost, dyn_model);

    // Constraints
    if (settings_.torque_limits)
    {
      ControlErrorResidual ctrl_fn = ControlErrorResidual(space.ndx(), Eigen::VectorXd::Zero(nu_));
      stm.addConstraint(ctrl_fn, BoxConstraint(settings_.umin, settings_.umax));
    }
    if (settings_.kinematics_limits)
    {
      StateErrorResidual state_fn = StateErrorResidual(space, nu_, space.neutral());
      std::vector<int> state_id;
      for (int i = 6; i < nv_; i++)
      {
        state_id.push_back(i);
      }
      FunctionSliceXpr state_slice = FunctionSliceXpr(state_fn, state_id);
      stm.addConstraint(state_slice, BoxConstraint(-settings_.qmax, -settings_.qmin));
    }

    for (auto const & name : model_handler_.getFeetNames())
    {
      if (settings_.force_size == 6 and contact_phase.at(name))
      {
        if (settings_.force_cone)
        {
          MultibodyWrenchConeResidual wrench_residual = MultibodyWrenchConeResidual(
            space.ndx(), model_handler_.getModel(), actuation_matrix_, cms, prox_settings_, name, settings_.mu,
            settings_.Lfoot, settings_.Wfoot);
          stm.addConstraint(wrench_residual, NegativeOrthant());
        }

        if (land_constraint.at(name))
        {
          FrameVelocityResidual velocity_residual = FrameVelocityResidual(
            space.ndx(), nu_, model_handler_.getModel(), Motion::Zero(), model_handler_.getFootId(name),
            pinocchio::LOCAL_WORLD_ALIGNED);
          stm.addConstraint(velocity_residual, EqualityConstraint());
        }
      }
      else if (settings_.force_size == 3 and contact_phase.at(name))
      {
        if (settings_.force_cone)
        {
          MultibodyFrictionConeResidual friction_residual = MultibodyFrictionConeResidual(
            space.ndx(), model_handler_.getModel(), actuation_matrix_, cms, prox_settings_, name, settings_.mu);
          stm.addConstraint(friction_residual, NegativeOrthant());
        }
        if (land_constraint.at(name))
        {
          std::vector<int> vel_id = {0, 1, 2};
          FrameVelocityResidual velocity_residual = FrameVelocityResidual(
            space.ndx(), nu_, model_handler_.getModel(), Motion::Zero(), model_handler_.getFootId(name),
            pinocchio::LOCAL_WORLD_ALIGNED);
          FunctionSliceXpr vel_slice = FunctionSliceXpr(velocity_residual, vel_id);
          stm.addConstraint(vel_slice, EqualityConstraint());

          std::vector<int> frame_id = {2};

          FrameTranslationResidual frame_residual = FrameTranslationResidual(
            space.ndx(), nu_, model_handler_.getModel(), contact_pose.at(name).translation(),
            model_handler_.getFootId(name));

          FunctionSliceXpr frame_slice = FunctionSliceXpr(frame_residual, frame_id);

          stm.addConstraint(frame_slice, EqualityConstraint());
        }
      }
    }

    return stm;
  }

  void FullDynamicsOCP::setReferencePoses(const std::size_t t, const std::map<std::string, pinocchio::SE3> & pose_refs)
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

  void
  FullDynamicsOCP::setReferencePose(const std::size_t t, const std::string & ee_name, const pinocchio::SE3 & pose_ref)
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

  void FullDynamicsOCP::setTerminalReferencePose(const std::string & ee_name, const pinocchio::SE3 & pose_ref)
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

  void
  FullDynamicsOCP::setReferenceForces(const std::size_t t, const std::map<std::string, Eigen::VectorXd> & force_refs)
  {
    CostStack * cs = getCostStack(t);
    if (force_refs.size() != model_handler_.getFeetNames().size())
    {
      throw std::runtime_error("force_refs size does not match number of end effectors");
    }
    for (auto ee_name : model_handler_.getFeetNames())
    {
      QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>(ee_name + "_force_cost");
      ContactForceResidual * cfr = qrc->getResidual<ContactForceResidual>();
      cfr->setReference(force_refs.at(ee_name));
    }
  }

  void
  FullDynamicsOCP::setReferenceForce(const std::size_t i, const std::string & ee_name, const ConstVectorRef & force_ref)
  {
    CostStack * cs = getCostStack(i);
    QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>(ee_name + "_force_cost");
    ContactForceResidual * cfr = qrc->getResidual<ContactForceResidual>();
    cfr->setReference(force_ref);
  }

  const pinocchio::SE3 FullDynamicsOCP::getReferencePose(const std::size_t t, const std::string & ee_name)
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

  const Eigen::VectorXd FullDynamicsOCP::getReferenceForce(const std::size_t t, const std::string & ee_name)
  {
    CostStack * cs = getCostStack(t);
    QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>(ee_name + "_force_cost");
    ContactForceResidual * cfr = qrc->getResidual<ContactForceResidual>();
    return cfr->getReference();
  }

  const Eigen::VectorXd FullDynamicsOCP::getVelocityBase(const std::size_t t)
  {
    CostStack * cs = getCostStack(t);
    QuadraticStateCost * qc = cs->getComponent<QuadraticStateCost>("state_cost");
    return qc->getTarget().segment(nq_, 6);
  }

  void FullDynamicsOCP::setVelocityBase(const std::size_t t, const ConstVectorRef & velocity_base)
  {
    if (velocity_base.size() != 6)
    {
      throw std::runtime_error("velocity_base size should be 6");
    }
    CostStack * cs = getCostStack(t);
    QuadraticStateCost * qc = cs->getComponent<QuadraticStateCost>("state_cost");
    x0_.segment(nq_, 6) = velocity_base;
    qc->setTarget(x0_);
  }

  const Eigen::VectorXd FullDynamicsOCP::getPoseBase(const std::size_t t)
  {
    CostStack * cs = getCostStack(t);
    QuadraticStateCost * qc = cs->getComponent<QuadraticStateCost>("state_cost");
    return qc->getTarget().head(7);
  };

  void FullDynamicsOCP::setPoseBase(const std::size_t t, const ConstVectorRef & pose_base)
  {
    if (pose_base.size() != 7)
    {
      throw std::runtime_error("pose_base size should be 7");
    }
    CostStack * cs = getCostStack(t);
    QuadraticStateCost * qc = cs->getComponent<QuadraticStateCost>("state_cost");
    x0_.head(7) = pose_base;
    qc->setTarget(x0_);
  }

  const Eigen::VectorXd FullDynamicsOCP::getProblemState(const RobotDataHandler & data_handler)
  {
    return data_handler.getState();
  }

  size_t FullDynamicsOCP::getContactSupport(const std::size_t t)
  {
    MultibodyConstraintFwdDynamics * ode =
      problem_->stages_[t]->getDynamics<IntegratorSemiImplEuler>()->getDynamics<MultibodyConstraintFwdDynamics>();

    return ode->constraint_models_.size();
  }

  std::vector<bool> FullDynamicsOCP::getContactState(const std::size_t t)
  {
    std::vector<bool> contact_state;
    MultibodyConstraintFwdDynamics * ode =
      problem_->stages_[t]->getDynamics<IntegratorSemiImplEuler>()->getDynamics<MultibodyConstraintFwdDynamics>();
    assert(ode != nullptr);
    for (auto name : model_handler_.getFeetNames())
    {
      std::size_t i;
      for (i = 0; i < ode->constraint_models_.size(); i++)
      {
        if (ode->constraint_models_[i].name == name)
        {
          contact_state.push_back(true);
          break;
        }
      }
      if (i == ode->constraint_models_.size())
      {
        contact_state.push_back(false);
      }
    }
    return contact_state;
  }

  CostStack FullDynamicsOCP::createTerminalCost()
  {
    auto ter_space = MultibodyPhaseSpace(model_handler_.getModel());
    auto term_cost = CostStack(ter_space, nu_);
    auto cent_mom =
      CentroidalMomentumResidual(ter_space.ndx(), nu_, model_handler_.getModel(), Eigen::VectorXd::Zero(6));

    term_cost.addCost("state_cost", QuadraticStateCost(ter_space, nu_, x0_, settings_.w_x));
    /* term_cost.addCost(
        "centroidal_cost",
        QuadraticResidualCost(ter_space, cent_mom, settings_.w_cent)); */

    return term_cost;
  }

  void FullDynamicsOCP::createTerminalConstraint(const Eigen::Vector3d & com_ref)
  {
    if (!problem_initialized_)
    {
      throw std::runtime_error("Create problem first!");
    }
    CenterOfMassTranslationResidual com_cstr =
      CenterOfMassTranslationResidual(ndx_, nu_, model_handler_.getModel(), com_ref);

    double tau = sqrt(com_ref[2] / 9.81);
    DCMPositionResidual dcm_cstr = DCMPositionResidual(ndx_, nu_, model_handler_.getModel(), com_ref, tau);

    problem_->addTerminalConstraint(dcm_cstr, EqualityConstraint());

    Motion v_ref = Motion::Zero();
    for (auto const & name : model_handler_.getFeetNames())
    {
      FrameVelocityResidual frame_vel = FrameVelocityResidual(
        ndx_, nu_, model_handler_.getModel(), v_ref, model_handler_.getFootId(name), pinocchio::LOCAL_WORLD_ALIGNED);
      if (settings_.force_size == 6)
        problem_->addTerminalConstraint(frame_vel, EqualityConstraint());
      else
      {
        std::vector<int> vel_id = {0, 1, 2};

        FunctionSliceXpr vel_slice = FunctionSliceXpr(frame_vel, vel_id);
        problem_->addTerminalConstraint(vel_slice, EqualityConstraint());
      }
    }

    terminal_constraint_ = true;
  }

  void FullDynamicsOCP::updateTerminalConstraint(const Eigen::Vector3d & com_ref)
  {
    if (terminal_constraint_)
    {
      DCMPositionResidual * CoMres = problem_->term_cstrs_.getConstraint<DCMPositionResidual>(0);

      CoMres->setReference(com_ref);
    }
  }

} // namespace simple_mpc
