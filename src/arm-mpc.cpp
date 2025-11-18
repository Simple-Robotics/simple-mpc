///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2025, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "simple-mpc/arm-mpc.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <chrono>

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
    solver_->reg_min = 1e-6;

    ee_name_ = problem->getSettings().ee_name;

    for (std::size_t i = 0; i < ocp_handler_->getProblem().numSteps(); i++)
    {
      xs_.push_back(x0_);
      us_.push_back(Eigen::VectorXd::Zero(model_handler.getModel().nv));
    }
    xs_.push_back(x0_);

    solver_->setup(ocp_handler_->getProblem());
    solver_->run(ocp_handler_->getProblem(), xs_, us_);

    xs_ = solver_->results_.xs;
    us_ = solver_->results_.us;
    Ks_ = solver_->results_.getCtrlFeedbacks();

    solver_->max_iters = settings_.max_iters;
    now_ = RESTING;
    reach_pose_ = pinocchio::SE3::Identity();
  }

  void ArmMPC::generateContactHorizon(const Eigen::Vector3d & contact_force)
  {
    for (std::size_t i = 0; i < ocp_handler_->getProblem().numSteps(); i++)
    {
      StageModel sm = StageModel(ocp_handler_->createStage(true, reach_pose_, true, contact_force));
      contact_horizon_.push_back(sm);
      contact_horizon_data_.push_back(sm.createData());

      StageModel sm_nocontact = StageModel(ocp_handler_->createStage(true, reach_pose_));
      no_contact_horizon_.push_back(sm_nocontact);
      no_contact_horizon_data_.push_back(sm_nocontact.createData());
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
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    solver_->run(ocp_handler_->getProblem(), xs_, us_);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference for run = "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    // Collect results
    xs_ = solver_->results_.xs;
    us_ = solver_->results_.us;
    Ks_ = solver_->results_.getCtrlFeedbacks();
  }

  void ArmMPC::recedeWithCycle()
  {
    std::size_t last_id = ocp_handler_->getSize() - 1;
    // rotate_vec_left(ocp_handler_->getProblem().stages_);
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (now_ == CONTACT)
    {
      // ocp_handler_->addContact(last_id);
      ocp_handler_->getProblem().replaceStageCircular(contact_horizon_[0]);
      solver_->cycleProblem(ocp_handler_->getProblem(), contact_horizon_data_[0]);
      rotate_vec_left(contact_horizon_);
      rotate_vec_left(contact_horizon_data_);
    }
    else if (now_ == REACHING or now_ == RESTING)
    {
      // ocp_handler_->removeContact(last_id);
      ocp_handler_->getProblem().replaceStageCircular(no_contact_horizon_[0]);
      solver_->cycleProblem(ocp_handler_->getProblem(), no_contact_horizon_data_[0]);
      rotate_vec_left(no_contact_horizon_);
      rotate_vec_left(no_contact_horizon_data_);
    }

    // shared_ptr<StageData> stage_data = ocp_handler_->getProblem().stages_.back()->createData();
    // solver_->cycleProblem(ocp_handler_->getProblem(), stage_data);
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // std::cout << "Time difference for cycleProblem = " << std::chrono::duration_cast<std::chrono::milliseconds>(end -
    // begin).count() << "[ms]" << std::endl;

    if (now_ == REACHING or now_ == CONTACT)
    {
      ocp_handler_->setWeight(last_id, "frame_cost", 1.0);
      ocp_handler_->setTerminalWeight("frame_cost", 1.0);
    }
    else
    {
      ocp_handler_->setWeight(last_id, "frame_cost", 0.0);
      ocp_handler_->setTerminalWeight("frame_cost", 0.0);
    }
  }

  void ArmMPC::updateTargetReference()
  {
    ocp_handler_->setReferencePose(ocp_handler_->getSize() - 1, reach_pose_);
    ocp_handler_->setTerminalReferencePose(reach_pose_);
    ocp_handler_->setReferenceState(ocp_handler_->getSize() - 1, x_reference_);
  }

  void ArmMPC::setReferencePose(const std::size_t t, const pinocchio::SE3 & pose_ref)
  {
    if (t < ocp_handler_->getSize() - 1)
      ocp_handler_->setReferencePose(t, pose_ref);
    else
      ocp_handler_->setTerminalReferencePose(pose_ref);
  }

  const pinocchio::SE3 ArmMPC::getReferencePose(const std::size_t t) const
  {
    pinocchio::SE3 pos_ref;
    if (t < ocp_handler_->getSize() - 1)
      pos_ref = ocp_handler_->getReferencePose(t);
    else
      pos_ref = ocp_handler_->getTerminalReferencePose();

    return pos_ref;
  }

  void ArmMPC::setReferenceForce(const std::size_t t, const Eigen::Vector3d & contact_force)
  {
    ocp_handler_->setReferenceForce(t, contact_force);
  }

  const Eigen::Vector3d ArmMPC::getReferenceForce(const std::size_t t) const
  {
    Eigen::Vector3d force_ref = ocp_handler_->getReferenceForce(t);

    return force_ref;
  }

  const Eigen::Vector3d ArmMPC::getContactForce(const std::size_t t)
  {
    ExplicitIntegratorData * int_data =
      dynamic_cast<ExplicitIntegratorData *>(&*solver_->workspace_.problem_data.stage_data[t]->dynamics_data);
    assert(int_data != nullptr);
    MultibodyConstraintFwdData * mc_data = dynamic_cast<MultibodyConstraintFwdData *>(&*int_data->continuous_data);
    assert(mc_data != nullptr);

    Eigen::Vector3d contact_forces = Eigen::Vector3d::Zero();
    if (mc_data->constraint_datas_.size() > 0)
    {
      contact_forces = mc_data->constraint_datas_[0].contact_force.linear();
    }

    return contact_forces;
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

  void ArmMPC::switchToReach(const pinocchio::SE3 & reach_pose)
  {
    now_ = REACHING;
    reach_pose_ = reach_pose;
  }

  void ArmMPC::switchToRest()
  {
    now_ = RESTING;
  }

  void ArmMPC::switchToContact(const pinocchio::SE3 & reach_pose)
  {
    now_ = CONTACT;
    reach_pose_ = reach_pose;
  }

} // namespace simple_mpc
