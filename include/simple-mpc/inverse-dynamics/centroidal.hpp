#pragma once

#include <simple-mpc/inverse-dynamics/kinodynamics.hpp>
#include <tsid/tasks/task-com-equality.hpp>

namespace simple_mpc
{

  class CentroidalID : public KinodynamicsID
  {
  public:
    typedef Eigen::VectorXd TargetContactForce;
    // typedef Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 6, 1> TargetContactForce;
    typedef PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::SE3) FeetPoseVector;
    typedef PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Motion) FeetVelocityVector;

    struct Settings : public KinodynamicsID::Settings
    {
      // Tasks gains
      double kp_com = 0.;
      double kp_feet_tracking = 0.;

      // Tasks weights
      double w_com = -1.;           // Disabled by default
      double w_feet_tracking = -1.; // Disabled by default
    };

    CentroidalID(const RobotModelHandler & model_handler, double control_dt, const Settings settings);
    CentroidalID(const CentroidalID &) = delete;

    void setTarget(
      const Eigen::Ref<const Eigen::Vector<double, 3>> & com_position,
      const Eigen::Ref<const Eigen::Vector<double, 3>> & com_velocity,
      const FeetPoseVector & feet_pose,
      const FeetVelocityVector & feet_velocity,
      const std::vector<bool> & contact_state_target,
      const std::vector<TargetContactForce> & f_target);

    using KinodynamicsID::getAccelerations;
    using KinodynamicsID::solve;

  private:
    using KinodynamicsID::KinodynamicsID;
    using KinodynamicsID::setTarget;

  public:
    // Order matters to be instantiated in the right order
    const Settings settings_;

  private:
    std::unique_ptr<tsid::tasks::TaskComEquality> comTask_;
    std::vector<tsid::tasks::TaskSE3Equality> trackingTasks_;
    tsid::trajectories::TrajectorySample sampleCom_;
    std::vector<tsid::trajectories::TrajectorySample> trackingSamples_;
    std::vector<bool> feet_tracked_;
  };

} // namespace simple_mpc
