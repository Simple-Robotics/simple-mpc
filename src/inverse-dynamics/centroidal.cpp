#include <simple-mpc/inverse-dynamics/centroidal.hpp>

namespace simple_mpc
{

  CentroidalID::CentroidalID(const RobotModelHandler & model_handler, double control_dt, const Settings settings)
  : KinodynamicsID(model_handler, control_dt, settings)
  , settings_(settings)
  {
    // Update base task to be only on the base orientation
    const Eigen::Vector<double, 6> kp_base{0., 0., 0., 1., 1., 1.};
    baseTask_->Kp(settings_.kp_base * kp_base);
    baseTask_->Kd(2.0 * baseTask_->Kp().cwiseSqrt());

    // Add the center of mass task
    comTask_ = std::make_shared<tsid::tasks::TaskComEquality>("task-com", robot_);
    comTask_->Kp(settings_.kp_com * Eigen::VectorXd::Ones(3));
    comTask_->Kd(2.0 * comTask_->Kp().cwiseSqrt());
    if (settings_.w_com > 0.)
      formulation_.addMotionTask(*comTask_, settings_.w_com, 1);

    sampleCom_ = tsid::trajectories::TrajectorySample(3);

    // Add foot tracking task
    const size_t n_contacts = model_handler_.getFeetNb();
    for (int i = 0; i < n_contacts; i++)
    {
      const std::string frame_name = model_handler.getFootFrameName(i);
      tsid::tasks::TaskSE3Equality & trackingTask =
        trackingTasks_.emplace_back("task-tracking-" + frame_name, robot_, frame_name);
      trackingTask.Kp(settings_.kp_feet_tracking * Eigen::VectorXd::Ones(6));
      trackingTask.Kd(2.0 * trackingTask.Kp().cwiseSqrt());
      // Do not add tasks ; will be done in setTarget depending on desired contacts.
      trackingSamples_.emplace_back(12, 6);
    }
  }

  void CentroidalID::setTarget(
    const Eigen::Ref<const Eigen::Vector<double, 3>> & com_position,
    const Eigen::Ref<const Eigen::Vector<double, 3>> & com_velocity,
    const FeetPoseVector & feet_pose,
    const FeetVelocityVector & feet_velocity,
    const std::vector<bool> & contact_state_target,
    const std::vector<TargetContactForce> & f_target)
  {
    const size_t nq = model_handler_.getModel().nq;
    const size_t nv = model_handler_.getModel().nv;

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
            active_tsid_contacts_[foot_nb] = false;
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
