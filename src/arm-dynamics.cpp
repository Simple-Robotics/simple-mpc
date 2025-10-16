#include "simple-mpc/arm-dynamics.hpp"
#include "simple-mpc/ocp-handler.hpp"

#include "aligator/modelling/dynamics/multibody-free-fwd.hpp"
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <aligator/modelling/multibody/frame-placement.hpp>
#include <aligator/modelling/multibody/frame-translation.hpp>
#include <aligator/modelling/multibody/frame-velocity.hpp>

namespace simple_mpc
{
  using namespace aligator;
  using MultibodyPhaseSpace = MultibodyPhaseSpace<double>;
  using MultibodyFreeFwdDynamics = dynamics::MultibodyFreeFwdDynamicsTpl<double>;
  using FramePlacementResidual = FramePlacementResidualTpl<double>;
  using FrameTranslationResidual = FrameTranslationResidualTpl<double>;
  using FrameVelocityResidual = FrameVelocityResidualTpl<double>;
  using IntegratorSemiImplEuler = dynamics::IntegratorSemiImplEulerTpl<double>;

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
  }

  StageModel ArmDynamicsOCP::createStage(const bool reaching, const Eigen::Vector3d & reach_pose)
  {

    auto space = MultibodyPhaseSpace(model_handler_.getModel());
    auto rcost = CostStack(space, nv_);

    rcost.addCost("state_cost", QuadraticStateCost(space, nv_, model_handler_.getReferenceState(), settings_.w_x));
    rcost.addCost("control_cost", QuadraticControlCost(space, Eigen::VectorXd::Zero(nv_), settings_.w_u));

    FrameTranslationResidual frame_residual =
      FrameTranslationResidual(space.ndx(), nv_, model_handler_.getModel(), reach_pose, ee_id_);
    if (reaching)
    {
      rcost.addCost(settings_.ee_name + "_cost", QuadraticResidualCost(space, frame_residual, settings_.w_frame));
    }
    else
    {
      rcost.addCost(settings_.ee_name + "_cost", QuadraticResidualCost(space, frame_residual, settings_.w_frame), 0.);
    }

    MultibodyFreeFwdDynamics ode = MultibodyFreeFwdDynamics(space, Eigen::MatrixXd::Identity(nv_, nv_));
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
      StateErrorResidual state_fn = StateErrorResidual(space, nv_, space.neutral());
      stm.addConstraint(state_fn, BoxConstraint(settings_.qmin, settings_.qmax));
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

  void ArmDynamicsOCP::setReferencePose(const std::size_t t, const Eigen::Vector3d & pose_ref)
  {
    CostStack * cs = getCostStack(t);
    QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>(settings_.ee_name + "_cost");
    FrameTranslationResidual * cfr = qrc->getResidual<FrameTranslationResidual>();
    cfr->setReference(pose_ref);
  }

  const Eigen::Vector3d ArmDynamicsOCP::getReferencePose(const std::size_t t)
  {
    CostStack * cs = getCostStack(t);
    QuadraticResidualCost * qrc = cs->getComponent<QuadraticResidualCost>(settings_.ee_name + "_cost");
    FrameTranslationResidual * cfr = qrc->getResidual<FrameTranslationResidual>();
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

  CostStack ArmDynamicsOCP::createTerminalCost()
  {
    auto ter_space = MultibodyPhaseSpace(model_handler_.getModel());
    auto term_cost = CostStack(ter_space, nv_);

    term_cost.addCost(
      "state_cost", QuadraticStateCost(ter_space, nv_, model_handler_.getReferenceState(), settings_.w_x));

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
