#include <simple-mpc/inverse-dynamics/centroidal-id.hpp>

namespace simple_mpc
{

  CentroidalID::CentroidalID(const RobotModelHandler & model_handler, double control_dt, const Settings settings)
  : KinodynamicsID(model_handler, control_dt, settings)
  , settings_(settings)
  {
    // Update base task to be only on the base orientation
    if (baseTask_)
    {
      const Eigen::Vector<double, 6> orientation_mask{0., 0., 0., 1., 1., 1.};
      baseTask_->Kp(settings_.kp_base * Eigen::VectorXd::Ones(6));
      baseTask_->Kd(2.0 * baseTask_->Kp().cwiseSqrt());
      baseTask_->setMask(orientation_mask);
      if (settings_.w_base > 0.)
      {
        // Task has changed size, need to be removed and added again for new size to be taken into account.
        formulation_.removeTask(baseTask_->name(), 0);
        formulation_.addMotionTask(*baseTask_, settings_.w_base, 1);
      }
    }

    // Add the center of mass task
    comTask_ = std::make_unique<tsid::tasks::TaskComEquality>("task-com", robot_);
    comTask_->Kp(settings_.kp_com * Eigen::VectorXd::Ones(3));
    comTask_->Kd(2.0 * comTask_->Kp().cwiseSqrt());
    if (settings_.w_com > 0.)
      formulation_.addMotionTask(*comTask_, settings_.w_com, 1);

    sampleCom_.resize(3);

    // Add foot tracking task
    const size_t n_contacts = model_handler_.getFeetNb();
    for (size_t i = 0; i < n_contacts; i++)
    {
      const RobotModelHandler::FootType foot_type = model_handler.getFootType(i);
      const std::string frame_name = model_handler.getFootFrameName(i);
      Eigen::Vector<double, 6> position_mask = Eigen::Vector<double, 6>::Ones();
      if (foot_type == RobotModelHandler::POINT)
        position_mask.tail<3>().setZero();
      else if (foot_type == RobotModelHandler::QUAD)
        position_mask.tail<3>().setOnes();
      else
        assert(false);
      tsid::tasks::TaskSE3Equality & trackingTask =
        trackingTasks_.emplace_back("task-tracking-" + frame_name, robot_, frame_name);
      trackingTask.Kp(settings_.kp_feet_tracking * Eigen::VectorXd::Ones(6));
      trackingTask.Kd(2.0 * trackingTask.Kp().cwiseSqrt());
      trackingTask.setMask(position_mask);
      // Do not add tasks ; will be done in setTarget depending on desired contacts.
      feet_tracked_.push_back(false);
      trackingSamples_.emplace_back(12, 6);
    }

    // By default initialize target in reference state
    const int nq = model_handler_.getModel().nq;
    const int nv = model_handler_.getModel().nv;
    const Eigen::VectorXd q_ref = model_handler.getReferenceState().head(nq);
    const Eigen::VectorXd v_ref = model_handler.getReferenceState().tail(nv);
    data_handler_.updateInternalData(q_ref, v_ref, false);
    const Eigen::Vector3d com_pos{data_handler_.getData().com[0]};
    const Eigen::Vector3d com_vel{0., 0., 0.};
    FeetPoseVector feet_pose(n_contacts);
    FeetVelocityVector feet_vel(n_contacts);
    std::vector<bool> feet_contact(n_contacts);
    std::vector<TargetContactForce> feet_force;
    for (size_t i = 0; i < n_contacts; i++)
    {
      // By default initialize all foot in contact with same amount of force
      feet_contact[i] = true;
      const RobotModelHandler::FootType foot_type = model_handler.getFootType(i);
      if (foot_type == RobotModelHandler::POINT)
        feet_force.push_back(TargetContactForce::Zero(3));
      else if (foot_type == RobotModelHandler::QUAD)
        feet_force.push_back(TargetContactForce::Zero(6));
      else
        assert(false);
      feet_force[i][2] = 9.81 * model_handler_.getMass() / static_cast<double>(n_contacts); // Weight on Z axis
      feet_pose[i] = data_handler_.getFootRefPose(i);
      feet_vel[i].setZero();
    }
    setTarget(com_pos, com_vel, feet_pose, feet_vel, feet_contact, feet_force);

    // Dry run to initialize solver data & output
    const tsid::solvers::HQPData & solver_data = formulation_.computeProblemData(0, q_ref, v_ref);
    last_solution_ = solver_.solve(solver_data);
  }

  void CentroidalID::setTarget(
    const Eigen::Ref<const Eigen::Vector<double, 3>> & com_position,
    const Eigen::Ref<const Eigen::Vector<double, 3>> & com_velocity,
    const FeetPoseVector & feet_pose,
    const FeetVelocityVector & feet_velocity,
    const std::vector<bool> & contact_state_target,
    const std::vector<TargetContactForce> & f_target)
  {
    const int nq = model_handler_.getModel().nq;
    const int nv = model_handler_.getModel().nv;

    // Set CoM target
    sampleCom_.setValue(com_position);
    sampleCom_.setDerivative(com_velocity);
    sampleCom_.setSecondDerivative(Eigen::Vector3d::Zero());
    comTask_->setReference(sampleCom_);

    // Set feet tracking
    if (settings_.w_feet_tracking > 0.)
    {
      for (std::size_t foot_nb = 0; foot_nb < model_handler_.getFeetNb(); foot_nb++)
      {
        const std::string & task_name{"task-tracking-" + model_handler_.getFootFrameName(foot_nb)};
        if (!contact_state_target[foot_nb])
        {
          tsid::tasks::TaskSE3Equality & task{trackingTasks_[foot_nb]};
          tsid::trajectories::TrajectorySample & sample{trackingSamples_[foot_nb]};

          // Add tracking task to tsid if necessary
          if (!feet_tracked_[foot_nb])
          {
            formulation_.addMotionTask(task, settings_.w_feet_tracking, 1);
            feet_tracked_[foot_nb] = true;
          }
          // Set foot reference
          tsid::math::SE3ToVector(feet_pose[foot_nb], sample.pos);
          sample.vel.head<3>() = feet_velocity[foot_nb].linear();
          sample.vel.tail<3>() = feet_velocity[foot_nb].angular();
          sample.acc.setZero();
          task.setReference(sample);
        }
        else
        {
          // remove task if necessary
          if (feet_tracked_[foot_nb])
          {
            formulation_.removeTask(task_name, 0);
            feet_tracked_[foot_nb] = false;
          }
        }
      }
    }

    // Set kinodynamics targets (will resize solver properly)
    KinodynamicsID::setTarget(
      model_handler_.getReferenceState().head(nq), model_handler_.getReferenceState().tail(nv),
      Eigen::VectorXd::Zero(nv), contact_state_target, f_target);
  }

} // namespace simple_mpc
