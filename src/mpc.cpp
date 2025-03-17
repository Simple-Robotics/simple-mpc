///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "simple-mpc/mpc.hpp"
#include "simple-mpc/foot-trajectory.hpp"
#include "simple-mpc/ocp-handler.hpp"
#include "simple-mpc/robot-handler.hpp"

namespace simple_mpc
{
  using namespace aligator;
  constexpr std::size_t maxiters = 100;

  MPC::MPC(const MPCSettings & settings, std::shared_ptr<OCPHandler> problem)
  : settings_(settings)
  , ocp_handler_(problem)
  {

    data_handler_ = std::make_shared<RobotDataHandler>(ocp_handler_->getModelHandler());
    data_handler_->updateInternalData(ocp_handler_->getModelHandler().getReferenceState(), true);
    std::map<std::string, Eigen::Vector3d> starting_poses;
    for (auto const & name : ocp_handler_->getModelHandler().getFeetNames())
    {
      starting_poses.insert({name, data_handler_->getFootPose(name).translation()});

      relative_feet_poses_.insert(
        {name, data_handler_->getBaseFramePose().inverse() * data_handler_->getFootPose(name)});
    }
    foot_trajectories_ = FootTrajectory(
      starting_poses, settings_.swing_apex, settings_.T_fly, settings_.T_contact, ocp_handler_->getSize());

    foot_trajectories_.updateApex(settings.swing_apex);
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

    ee_names_ = ocp_handler_->getModelHandler().getFeetNames();
    Eigen::VectorXd force_ref(ocp_handler_->getReferenceForce(0, ocp_handler_->getModelHandler().getFootName(0)));

    std::map<std::string, bool> contact_states;
    std::map<std::string, bool> land_constraint;
    std::map<std::string, pinocchio::SE3> contact_poses;
    std::map<std::string, Eigen::VectorXd> force_map;

    for (auto const & name : ee_names_)
    {
      contact_states.insert({name, true});
      land_constraint.insert({name, false});
      contact_poses.insert({name, data_handler_->getFootPose(name)});
      force_map.insert({name, force_ref});
    }

    for (std::size_t i = 0; i < ocp_handler_->getProblem().numSteps(); i++)
    {
      xs_.push_back(x0_);
      us_.push_back(ocp_handler_->getReferenceControl(0));

      std::shared_ptr<StageModel> sm = std::make_shared<StageModel>(
        ocp_handler_->createStage(contact_states, contact_poses, force_map, land_constraint));
      standing_horizon_.push_back(sm);
      standing_horizon_data_.push_back(sm->createData());
    }
    xs_.push_back(x0_);

    solver_->setup(ocp_handler_->getProblem());
    solver_->run(ocp_handler_->getProblem(), xs_, us_);

    xs_ = solver_->results_.xs;
    us_ = solver_->results_.us;
    Ks_ = solver_->results_.getCtrlFeedbacks();

    solver_->max_iters = settings_.max_iters;

    com0_ = data_handler_->getData().com[0];
    now_ = WALKING;

    velocity_base_.setZero();
    next_pose_.setZero();
    twist_vect_.setZero();
  }

  void MPC::generateCycleHorizon(const std::vector<std::map<std::string, bool>> & contact_states)
  {
    contact_states_ = contact_states;
    // Guarantee that cycle horizon size is higher than problem size
    int m = int(ocp_handler_->getProblem().numSteps()) / int(contact_states.size());
    for (int i = 0; i < m; i++)
    {
      std::vector<std::map<std::string, bool>> copy_vec = contact_states;
      contact_states_.insert(contact_states_.end(), copy_vec.begin(), copy_vec.end());
    }

    // Generate contact switch timings
    for (auto const & name : ee_names_)
    {
      foot_takeoff_times_.insert({name, std::vector<int>()});
      foot_land_times_.insert({name, std::vector<int>()});
      for (size_t i = 1; i < contact_states_.size(); i++)
      {
        if (!contact_states_[i].at(name) and contact_states_[i - 1].at(name))
        {
          foot_takeoff_times_.at(name).push_back((int)(i + ocp_handler_->getSize()));
        }
        if (contact_states_[i].at(name) and !contact_states_[i - 1].at(name))
        {
          foot_land_times_.at(name).push_back((int)(i + ocp_handler_->getSize()));
        }
      }
      if (contact_states_.back().at(name) and !contact_states_[0].at(name))
        foot_takeoff_times_.at(name).push_back((int)(contact_states_.size() - 1 + ocp_handler_->getSize()));
      if (!contact_states_.back().at(name) and contact_states_[0].at(name))
        foot_land_times_.at(name).push_back((int)(contact_states_.size() - 1 + ocp_handler_->getSize()));
    }
    std::map<std::string, bool> previous_contacts;
    for (auto const & name : ee_names_)
    {
      previous_contacts.insert({name, true});
    }

    // Generate the model stages for cycle horizon
    for (auto const & state : contact_states_)
    {
      int active_contacts = 0;
      for (auto const & contact : state)
      {
        if (contact.second)
          active_contacts += 1;
      }

      Eigen::VectorXd force_ref(ocp_handler_->getReferenceForce(0, ocp_handler_->getModelHandler().getFootName(0)));
      Eigen::VectorXd force_zero(ocp_handler_->getReferenceForce(0, ocp_handler_->getModelHandler().getFootName(0)));
      force_ref.setZero();
      force_zero.setZero();
      force_ref[2] = settings_.support_force / active_contacts;

      std::map<std::string, pinocchio::SE3> contact_poses;
      std::map<std::string, Eigen::VectorXd> force_map;

      for (auto const & name : ee_names_)
      {
        contact_poses.insert({name, data_handler_->getFootPose(name)});
        if (state.at(name))
          force_map.insert({name, force_ref});
        else
          force_map.insert({name, force_zero});
      }
      std::map<std::string, bool> land_contacts;
      for (auto const & name : ee_names_)
      {
        if (!previous_contacts.at(name) and state.at(name))
        {
          land_contacts.insert({name, true});
        }
        else
        {
          land_contacts.insert({name, false});
        }
      }

      std::shared_ptr<StageModel> sm =
        std::make_shared<StageModel>(ocp_handler_->createStage(state, contact_poses, force_map, land_contacts));
      cycle_horizon_.push_back(sm);
      cycle_horizon_data_.push_back(sm->createData());
      previous_contacts = state;
    }
  }

  void MPC::iterate(const ConstVectorRef & x)
  {

    data_handler_->updateInternalData(x, false);

    // Recede all horizons
    recedeWithCycle();

    // Update the feet and CoM references
    updateStepTrackerReferences();

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

  void MPC::recedeWithCycle()
  {
    if (now_ == WALKING or ocp_handler_->getContactSupport(ocp_handler_->getSize() - 1) < ee_names_.size())
    {

      ocp_handler_->getProblem().replaceStageCircular(*cycle_horizon_[0]);
      solver_->cycleProblem(ocp_handler_->getProblem(), cycle_horizon_data_[0]);

      rotate_vec_left(cycle_horizon_);
      rotate_vec_left(cycle_horizon_data_);
      rotate_vec_left(contact_states_);
      for (auto const & name : ee_names_)
      {
        if (
          !contact_states_[contact_states_.size() - 1].at(name)
          and contact_states_[contact_states_.size() - 2].at(name))
          foot_takeoff_times_.at(name).push_back((int)(contact_states_.size() + ocp_handler_->getSize()));
        if (
          contact_states_[contact_states_.size() - 1].at(name)
          and !contact_states_[contact_states_.size() - 2].at(name))
          foot_land_times_.at(name).push_back((int)(contact_states_.size() + ocp_handler_->getSize()));
      }
      updateCycleTiming(false);
    }
    else
    {
      ocp_handler_->getProblem().replaceStageCircular(*standing_horizon_[0]);
      solver_->cycleProblem(ocp_handler_->getProblem(), standing_horizon_data_[0]);

      rotate_vec_left(standing_horizon_);
      rotate_vec_left(standing_horizon_data_);

      updateCycleTiming(true);
    }
  }

  void MPC::updateCycleTiming(const bool updateOnlyHorizon)
  {
    for (auto const & name : ee_names_)
    {
      for (size_t i = 0; i < foot_land_times_.at(name).size(); i++)
      {
        if (!updateOnlyHorizon or foot_land_times_.at(name)[i] < (int)ocp_handler_->getSize())
          foot_land_times_.at(name)[i] -= 1;
      }
      if (!foot_land_times_.at(name).empty() and foot_land_times_.at(name)[0] < 0)
        foot_land_times_.at(name).erase(foot_land_times_.at(name).begin());

      for (size_t i = 0; i < foot_takeoff_times_.at(name).size(); i++)
        if (!updateOnlyHorizon or foot_takeoff_times_.at(name)[i] < (int)ocp_handler_->getSize())
        {
          foot_takeoff_times_.at(name)[i] -= 1;
        }
      if (!foot_takeoff_times_.at(name).empty() and foot_takeoff_times_.at(name)[0] < 0)
        foot_takeoff_times_.at(name).erase(foot_takeoff_times_.at(name).begin());
    }
  }

  void MPC::updateStepTrackerReferences()
  {
    for (auto const & name : ee_names_)
    {
      int foot_land_time = -1;
      if (!foot_land_times_.at(name).empty())
        foot_land_time = foot_land_times_.at(name)[0];

      bool update = true;
      if (foot_land_time < settings_.T_fly)
        update = false;

      // Use the Raibert heuristics to compute the next foot pose
      twist_vect_[0] =
        -(data_handler_->getRefFootPose(name).translation()[1] - data_handler_->getBaseFramePose().translation()[1]);
      twist_vect_[1] =
        data_handler_->getRefFootPose(name).translation()[0] - data_handler_->getBaseFramePose().translation()[0];
      next_pose_.head(2) = data_handler_->getRefFootPose(name).translation().head(2);
      next_pose_.head(2) += (velocity_base_.head(2) + velocity_base_[5] * twist_vect_)
                            * (settings_.T_fly + settings_.T_contact) * settings_.timestep;
      next_pose_[2] = data_handler_->getFootPose(name).translation()[2];

      foot_trajectories_.updateTrajectory(
        update, foot_land_time, data_handler_->getFootPose(name).translation(), next_pose_, name);
      pinocchio::SE3 pose = pinocchio::SE3::Identity();
      for (unsigned long time = 0; time < ocp_handler_->getSize(); time++)
      {
        pose.translation() = foot_trajectories_.getReference(name)[time];
        setReferencePose(time, name, pose);
      }
    }

    ocp_handler_->setReferenceState(ocp_handler_->getSize() - 1, x_reference_);
    ocp_handler_->setVelocityBase(ocp_handler_->getSize() - 1, velocity_base_);

    Eigen::Vector3d com_ref;
    com_ref << 0, 0, 0;
    for (auto const & name : ee_names_)
    {
      com_ref += foot_trajectories_.getReference(name).back();
    }
    com_ref /= (double)ee_names_.size();
    com_ref[2] += com0_[2];

    ocp_handler_->updateTerminalConstraint(com_ref);
  }

  void MPC::setReferencePose(const std::size_t t, const std::string & ee_name, const pinocchio::SE3 & pose_ref)
  {
    ocp_handler_->setReferencePose(t, ee_name, pose_ref);
  }

  void MPC::setTerminalReferencePose(const std::string & ee_name, const pinocchio::SE3 & pose_ref)
  {
    ocp_handler_->setTerminalReferencePose(ee_name, pose_ref);
  }

  const pinocchio::SE3 MPC::getReferencePose(const std::size_t t, const std::string & ee_name) const
  {
    return ocp_handler_->getReferencePose(t, ee_name);
  }

  TrajOptProblem & MPC::getTrajOptProblem()
  {
    return ocp_handler_->getProblem();
  }

  const ConstVectorRef MPC::getStateDerivative(const std::size_t t)
  {
    ExplicitIntegratorData * int_data =
      dynamic_cast<ExplicitIntegratorData *>(&*solver_->workspace_.problem_data.stage_data[t]->dynamics_data);
    assert(int_data != nullptr);
    return int_data->continuous_data->xdot_;
  }

  const Eigen::VectorXd MPC::getContactForces(const std::size_t t)
  {
    Eigen::VectorXd contact_forces;
    contact_forces.resize(3 * (long)ee_names_.size());

    ExplicitIntegratorData * int_data =
      dynamic_cast<ExplicitIntegratorData *>(&*solver_->workspace_.problem_data.stage_data[t]->dynamics_data);
    assert(int_data != nullptr);
    MultibodyConstraintFwdData * mc_data = dynamic_cast<MultibodyConstraintFwdData *>(&*int_data->continuous_data);
    assert(mc_data != nullptr);

    std::vector<bool> contact_state = ocp_handler_->getContactState(t);

    size_t force_id = 0;
    for (size_t i = 0; i < contact_state.size(); i++)
    {
      if (contact_state[i])
      {
        contact_forces.segment((long)i * 3, 3) = mc_data->constraint_datas_[force_id].contact_force.linear();
        force_id += 1;
      }
      else
      {
        contact_forces.segment((long)i * 3, 3).setZero();
      }
    }
    return contact_forces;
  }

  void MPC::switchToWalk(const Vector6d & velocity_base)
  {
    now_ = WALKING;
    velocity_base_ = velocity_base;
  }

  void MPC::switchToStand()
  {
    now_ = STANDING;
    velocity_base_.setZero();
  }

} // namespace simple_mpc
