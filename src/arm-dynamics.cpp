#include "simple-mpc/arm-dynamics.hpp"
#include "simple-mpc/ocp-handler.hpp"

#include "aligator/modelling/dynamics/multibody-free-fwd.hpp"
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <aligator/modelling/multibody/contact-force.hpp>
#include <aligator/modelling/multibody/frame-placement.hpp>
#include <aligator/modelling/multibody/frame-translation.hpp>
#include <aligator/modelling/multibody/frame-velocity.hpp>
#include <pinocchio/multibody/fwd.hpp>

namespace simple_mpc
{
  using namespace aligator;
  using MultibodyPhaseSpace = MultibodyPhaseSpace<double>;
  using MultibodyFreeFwdDynamics = dynamics::MultibodyFreeFwdDynamicsTpl<double>;
  using FramePlacementResidual = FramePlacementResidualTpl<double>;
  using FrameVelocityResidual = FrameVelocityResidualTpl<double>;
  using IntegratorSemiImplEuler = dynamics::IntegratorSemiImplEulerTpl<double>;
  using ContactForceResidual = ContactForceResidualTpl<double>;

  ArmDynamicsOCP::ArmDynamicsOCP(const ArmDynamicsSettings & settings, const RobotModelHandler & model_handler)
  : settings_(settings)
  , model_handler_(model_handler)
  , problem_(nullptr)
  {
    x0_ = model_handler_.getReferenceState();
    nq_ = model_handler.getModel().nq;
    nv_ = model_handler.getModel().nv;
    ndx_ = 2 * model_handler.getModel().nv;

    ee_id_ = model_handler_.getModel().getFrameId(settings_.ee_name);

    prox_settings_ = ProximalSettings(1e-9, 1e-10, 10);
    actuation_matrix_.resize(nv_, nv_);
    actuation_matrix_.setIdentity();

    auto joint_ids = model_handler_.getModel().frames[ee_id_].parentJoint;
    pinocchio::SE3 pl1 = model_handler_.getModel().frames[ee_id_].placement;
    pinocchio::SE3 pl2 = pinocchio::SE3::Identity();
    pinocchio::RigidConstraintModel constraint_model = pinocchio::RigidConstraintModel(
      pinocchio::ContactType::CONTACT_3D, model_handler_.getModel(), joint_ids, pl1, 0, pl2,
      pinocchio::LOCAL_WORLD_ALIGNED);
    constraint_model.corrector.Kp = settings.Kp_correction;
    constraint_model.corrector.Kd = settings.Kd_correction;
    constraint_model.name = "hand_contact";
    constraint_models_.push_back(constraint_model);
  }

  StageModel ArmDynamicsOCP::createStage(
    const bool reaching,
    const pinocchio::SE3 & reach_pose,
    const bool is_contact,
    const Eigen::Vector3d & contact_force)
  {

    auto space = MultibodyPhaseSpace(model_handler_.getModel());
    auto rcost = CostStack(space, nv_);

    rcost.addCost("state_cost", QuadraticStateCost(space, nv_, x0_, settings_.w_x));
    rcost.addCost("control_cost", QuadraticControlCost(space, Eigen::VectorXd::Zero(nv_), settings_.w_u));

    FramePlacementResidual frame_residual =
      FramePlacementResidual(space.ndx(), nv_, model_handler_.getModel(), reach_pose, ee_id_);
    if (reaching)
    {
      rcost.addCost("frame_cost", QuadraticResidualCost(space, frame_residual, settings_.w_frame));
    }
    else
    {
      rcost.addCost("frame_cost", QuadraticResidualCost(space, frame_residual, settings_.w_frame), 0.);
    }

    pinocchio::context::RigidConstraintModelVector cms;
    if (is_contact)
    {
      cms.push_back(constraint_models_[0]);

      ContactForceResidual frame_force = ContactForceResidual(
        space.ndx(), model_handler_.getModel(), actuation_matrix_, cms, prox_settings_, contact_force, "hand_contact");
      rcost.addCost("hand_force_cost", QuadraticResidualCost(space, frame_force, settings_.w_forces));
    }

    MultibodyConstraintFwdDynamics ode = MultibodyConstraintFwdDynamics(space, actuation_matrix_, cms, prox_settings_);
    // MultibodyFreeFwdDynamics ode = MultibodyFreeFwdDynamics(space, Eigen::MatrixXd::Identity(nv_, nv_));
    IntegratorSemiImplEuler dyn_model = IntegratorSemiImplEuler(ode, settings_.timestep);

    StageModel stm = StageModel(rcost, dyn_model);

    // Constraints
    if (settings_.torque_limits)
    {
      ControlErrorResidual ctrl_fn = ControlErrorResidual(space.ndx(), Eigen::VectorXd::Zero(nv_));
      stm.addConstraint(ctrl_fn, BoxConstraint(settings_.umin, settings_.umax));
    }
    if (settings_.kinematics_limits)
    {
      std::vector<int> state_id;
      for (int i = 0; i < nv_; i++)
      {
        state_id.push_back(i);
      }
      StateErrorResidual state_fn = StateErrorResidual(space, nv_, space.neutral());
      FunctionSliceXpr state_slice = FunctionSliceXpr(state_fn, state_id);
      stm.addConstraint(state_slice, BoxConstraint(settings_.qmin, settings_.qmax));
    }

    return stm;
  }

  CostStack * ArmDynamicsOCP::getCostStack(std::size_t t)
  {
    if (t >= getSize())
    {
      throw std::runtime_error("Stage index exceeds stage vector size");
    }
    CostStack * cs = dynamic_cast<CostStack *>(&*problem_->stages_[t]->cost_);

    return cs;
  }

  MultibodyConstraintFwdDynamics * ArmDynamicsOCP::getDynamics(std::size_t t)
  {
    if (t >= getSize())
    {
      throw std::runtime_error("Stage index exceeds stage vector size");
    }
    MultibodyConstraintFwdDynamics * ode =
      problem_->stages_[t]->getDynamics<IntegratorSemiImplEuler>()->getDynamics<MultibodyConstraintFwdDynamics>();

    return ode;
  }

  CostStack * ArmDynamicsOCP::getTerminalCostStack()
  {
    CostStack * cs = dynamic_cast<CostStack *>(&*problem_->term_cost_);

    return cs;
  }

  void ArmDynamicsOCP::setReferencePose(const std::size_t t, const pinocchio::SE3 & pose_ref)
  {
    CostStack * cs = getCostStack(t);
    QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>("frame_cost");
    FramePlacementResidual * cfr = qrc->getResidual<FramePlacementResidual>();
    cfr->setReference(pose_ref);
  }

  const pinocchio::SE3 ArmDynamicsOCP::getReferencePose(const std::size_t t)
  {
    CostStack * cs = getCostStack(t);
    QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>("frame_cost");
    FramePlacementResidual * cfr = qrc->getResidual<FramePlacementResidual>();
    pinocchio::SE3 ref = cfr->getReference();

    return ref;
  }

  void ArmDynamicsOCP::setTerminalReferencePose(const pinocchio::SE3 & pose_ref)
  {
    CostStack * cs = getTerminalCostStack();
    QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>("frame_cost");
    FramePlacementResidual * cfr = qrc->getResidual<FramePlacementResidual>();
    cfr->setReference(pose_ref);
  }

  const pinocchio::SE3 ArmDynamicsOCP::getTerminalReferencePose()
  {
    CostStack * cs = getTerminalCostStack();
    QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>("frame_cost");
    FramePlacementResidual * cfr = qrc->getResidual<FramePlacementResidual>();
    pinocchio::SE3 ref = cfr->getReference();

    return ref;
  }

  void ArmDynamicsOCP::setReferenceForce(const std::size_t t, const Eigen::Vector3d & force_ref)
  {
    CostStack * cs = getCostStack(t);
    QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>("hand_force_cost");
    ContactForceResidual * cfr = qrc->getResidual<ContactForceResidual>();
    cfr->setReference(force_ref);
  }

  const Eigen::Vector3d ArmDynamicsOCP::getReferenceForce(const std::size_t t)
  {
    CostStack * cs = getCostStack(t);
    QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>("hand_force_cost");
    ContactForceResidual * cfr = qrc->getResidual<ContactForceResidual>();
    Eigen::Vector3d ref = cfr->getReference();

    return ref;
  }

  const Eigen::VectorXd ArmDynamicsOCP::getProblemState(const RobotDataHandler & data_handler)
  {
    return data_handler.getState();
  }

  void ArmDynamicsOCP::setReferenceState(const std::size_t t, const ConstVectorRef & x_ref)
  {
    assert(x_ref.size() == nq_ + nv_ && "x_ref not of the right size");
    CostStack * cs = getCostStack(t);
    QuadraticStateCost * qc = cs->getComponent<QuadraticStateCost>("state_cost");
    qc->setTarget(x_ref);
  }

  const ConstVectorRef ArmDynamicsOCP::getReferenceState(const std::size_t t)
  {
    CostStack * cs = getCostStack(t);
    QuadraticStateCost * qc = cs->getComponent<QuadraticStateCost>("state_cost");
    return qc->getTarget();
  }

  void ArmDynamicsOCP::setTerminalWeight(const std::string key, double weight)
  {
    CostStack * cs = getTerminalCostStack();
    cs->setWeight(key, weight);
  }

  double ArmDynamicsOCP::getTerminalWeight(const std::string key)
  {
    CostStack * cs = getTerminalCostStack();
    return cs->getWeight(key);
  }

  void ArmDynamicsOCP::setWeight(const std::size_t t, const std::string key, double weight)
  {
    CostStack * cs = getCostStack(t);
    cs->setWeight(key, weight);
  }

  double ArmDynamicsOCP::getWeight(const std::size_t t, const std::string key)
  {
    CostStack * cs = getCostStack(t);
    return cs->getWeight(key);
  }

  void ArmDynamicsOCP::removeContact(const std::size_t t)
  {
    MultibodyConstraintFwdDynamics * md = getDynamics(t);

    if (md->constraint_models_.size() > 0)
    {
      md->constraint_models_.pop_back();
    }
  }

  void ArmDynamicsOCP::addContact(const std::size_t t)
  {
    MultibodyConstraintFwdDynamics * md = getDynamics(t);

    if (md->constraint_models_.size() == 0)
    {
      md->constraint_models_.push_back(constraint_models_[0]);
    }
  }

  CostStack ArmDynamicsOCP::createTerminalCost()
  {
    auto ter_space = MultibodyPhaseSpace(model_handler_.getModel());
    auto term_cost = CostStack(ter_space, nv_);

    term_cost.addCost(
      "state_cost", QuadraticStateCost(ter_space, nv_, model_handler_.getReferenceState(), settings_.w_x));

    FramePlacementResidual frame_residual =
      FramePlacementResidual(ter_space.ndx(), nv_, model_handler_.getModel(), pinocchio::SE3::Identity(), ee_id_);
    term_cost.addCost("frame_cost", QuadraticResidualCost(ter_space, frame_residual, settings_.w_frame), 0.0);

    return term_cost;
  }

  void ArmDynamicsOCP::createProblem(const ConstVectorRef & x0, const size_t horizon)
  {
    std::vector<xyz::polymorphic<StageModel>> stage_models;
    for (std::size_t i = 0; i < horizon; i++)
    {
      StageModel stage = createStage();
      stage_models.push_back(std::move(stage));
    }

    problem_ = std::make_unique<TrajOptProblem>(x0, std::move(stage_models), createTerminalCost());
    problem_initialized_ = true;
  }

} // namespace simple_mpc
