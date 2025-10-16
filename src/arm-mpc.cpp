///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2025, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "simple-mpc/arm-mpc.hpp"

namespace simple_mpc
{
  using namespace aligator;
  constexpr std::size_t maxiters = 100;

  ArmMPC::ArmMPC(const ArmMPCSettings & settings, std::shared_ptr<ArmDynamicsOCP> problem)
  : settings_(settings)
  , ocp_handler_(problem)
  {

    const RobotModelHandler & model_handler = ocp_handler_->getModelHandler();
    data_handler_ = std::make_shared<RobotDataHandler>(model_handler);
    data_handler_->updateInternalData(model_handler.getReferenceState(), true);

    x0_ = ocp_handler_->getProblemState(*data_handler_);
    x_reference_ = ocp_handler_->getReferenceState(0);

    solver_ = std::make_unique<SolverProxDDP>(settings_.TOL, settings_.mu_init, maxiters, aligator::QUIET);
    solver_->rollout_type_ = aligator::RolloutType::LINEAR;

    if (settings_.num_threads > 1)
    {
      solver_->linear_solver_choice = aligator::LQSolverChoice::PARALLEL;
      solver_->setNumThreads(settings_.num_threads);
    }
    else
      solver_->linear_solver_choice = aligator::LQSolverChoice::SERIAL;
    solver_->force_initial_condition_ = true;
    // solver_->reg_min = 1e-6;

    ee_name_ = problem->getSettings().ee_name;

    for (std::size_t i = 0; i < ocp_handler_->getProblem().numSteps(); i++)
    {
      xs_.push_back(x0_);
      us_.push_back(Eigen::VectorXd::Zero(model_handler.getModel().nv));

      std::shared_ptr<StageModel> sm = std::make_shared<StageModel>(ocp_handler_->createStage());
      rest_horizon_.push_back(sm);
      rest_horizon_data_.push_back(sm->createData());
    }
    xs_.push_back(x0_);

    solver_->setup(ocp_handler_->getProblem());
    solver_->run(ocp_handler_->getProblem(), xs_, us_);

    /*xs_ = solver_->results_.xs;
    us_ = solver_->results_.us;
    Ks_ = solver_->results_.getCtrlFeedbacks();

    solver_->max_iters = settings_.max_iters; */
  }

  void ArmMPC::generateReachHorizon(const Eigen::Vector3d & reach_pose)
  {
    reach_pose_ = reach_pose;
    // Generate the model stages for cycle horizon
    for (std::size_t i = 0; i < ocp_handler_->getProblem().numSteps(); i++)
    {

      std::shared_ptr<StageModel> sm = std::make_shared<StageModel>(ocp_handler_->createStage(true, reach_pose));
      reach_horizon_.push_back(sm);
      reach_horizon_data_.push_back(sm->createData());
    }
  }

  void ArmMPC::iterate(const ConstVectorRef & x)
  {

    data_handler_->updateInternalData(x, false);

    // Recede all horizons
    recedeWithCycle();

    // Update the feet and CoM references
    updateTargetReference();

    // Recede previous solutions
    x0_ << ocp_handler_->getProblemState(*data_handler_);
    xs_.erase(xs_.begin());
    xs_[0] = x0_;
    xs_.push_back(xs_.back());

    us_.erase(us_.begin());
    us_.push_back(us_.back());

    ocp_handler_->getProblem().setInitState(x0_);

    // Run solver
    solver_->run(ocp_handler_->getProblem(), xs_, us_);

    // Collect results
    xs_ = solver_->results_.xs;
    us_ = solver_->results_.us;
    Ks_ = solver_->results_.getCtrlFeedbacks();
  }

  void ArmMPC::recedeWithCycle()
  {
    if (now_ == REACHING)
    {
      ocp_handler_->getProblem().replaceStageCircular(*reach_horizon_[0]);
      solver_->cycleProblem(ocp_handler_->getProblem(), reach_horizon_data_[0]);

      rotate_vec_left(reach_horizon_);
      rotate_vec_left(reach_horizon_data_);
    }
    else
    {
      ocp_handler_->getProblem().replaceStageCircular(*rest_horizon_[0]);
      solver_->cycleProblem(ocp_handler_->getProblem(), rest_horizon_data_[0]);

      rotate_vec_left(rest_horizon_);
      rotate_vec_left(rest_horizon_data_);
    }
  }

  void ArmMPC::updateTargetReference()
  {
    ocp_handler_->setReferencePose(ocp_handler_->getSize() - 1, reach_pose_);
    ocp_handler_->setReferenceState(ocp_handler_->getSize() - 1, x_reference_);
  }

  const Eigen::Vector3d ArmMPC::getReferencePose(const std::size_t t) const
  {
    return ocp_handler_->getReferencePose(t);
  }

  TrajOptProblem & ArmMPC::getTrajOptProblem()
  {
    return ocp_handler_->getProblem();
  }

  const ConstVectorRef ArmMPC::getStateDerivative(const std::size_t t)
  {
    ExplicitIntegratorData * int_data =
      dynamic_cast<ExplicitIntegratorData *>(&*solver_->workspace_.problem_data.stage_data[t]->dynamics_data);
    assert(int_data != nullptr);
    return int_data->continuous_data->xdot_;
  }

  void ArmMPC::switchToReach(const Eigen::Vector3d & reach_pose)
  {
    now_ = REACHING;
    reach_pose_ = reach_pose;
  }

  void ArmMPC::switchToRest()
  {
    now_ = RESTING;
  }

} // namespace simple_mpc
