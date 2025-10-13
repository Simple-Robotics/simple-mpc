#pragma once

#include <simple-mpc/robot-handler.hpp>
#include <tsid/contacts/contact-point.hpp>
#include <tsid/formulations/inverse-dynamics-formulation-acc-force.hpp>
#include <tsid/solvers/solver-proxqp.hpp>
#include <tsid/solvers/utils.hpp>
#include <tsid/tasks/task-actuation-bounds.hpp>
#include <tsid/tasks/task-joint-posVelAcc-bounds.hpp>
#include <tsid/tasks/task-joint-posture.hpp>
#include <tsid/tasks/task-se3-equality.hpp>
#include <tsid/trajectories/trajectory-base.hpp>

namespace simple_mpc
{

  class KinodynamicsID
  {
  public:
    typedef Eigen::VectorXd TargetContactForce;

    struct Settings
    {

      // Physical quantities
      double friction_coefficient = 0.6;

      double contact_weight_ratio_max =
        10.0; // Max force for one foot contact (express as a multiple of the robot weight)

      double contact_weight_ratio_min =
        0.01; // Min force for one foot contact (express as a multiple of the robot weight)

      // Tasks gains
      double kp_base = 0.;
      double kp_posture = 0.;
      double kp_contact = 0.;

      // Tasks weights
      double w_base = -1.;           // Disabled by default
      double w_posture = -1.;        // Disabled by default
      double w_contact_motion = -1.; // Disabled by default
      double w_contact_force = -1.;  // Disabled by default

      ///< Are the contact motion = 0 handled as a hard contraint (true) or a cost (if false)
      bool contact_motion_equality = false;
    };

    KinodynamicsID(const simple_mpc::RobotModelHandler & model_handler, double control_dt, const Settings settings);
    KinodynamicsID(const KinodynamicsID &) = delete;

    void setTarget(
      const Eigen::Ref<const Eigen::VectorXd> & q_target,
      const Eigen::Ref<const Eigen::VectorXd> & v_target,
      const Eigen::Ref<const Eigen::VectorXd> & a_target,
      const std::vector<bool> & contact_state_target,
      const std::vector<TargetContactForce> & f_target);

    void solve(
      const double t,
      const Eigen::Ref<const Eigen::VectorXd> & q_meas,
      const Eigen::Ref<const Eigen::VectorXd> & v_meas,
      Eigen::Ref<Eigen::VectorXd> tau_res);

    void getAccelerations(Eigen::Ref<Eigen::VectorXd> a);

  public:
    // Order matters to be instantiated in the right order
    const Settings settings_;
    const simple_mpc::RobotModelHandler & model_handler_;

  protected:
    // Order still matters here
    simple_mpc::RobotDataHandler data_handler_;
    tsid::robots::RobotWrapper robot_;
    tsid::InverseDynamicsFormulationAccForce formulation_;

    std::vector<bool> active_tsid_contacts_;
    std::vector<std::unique_ptr<tsid::contacts::ContactBase>> tsid_contacts;
    std::unique_ptr<tsid::tasks::TaskJointPosture> postureTask_;
    std::unique_ptr<tsid::tasks::TaskSE3Equality> baseTask_;
    std::unique_ptr<tsid::tasks::TaskJointPosVelAccBounds> boundsTask_;
    std::unique_ptr<tsid::tasks::TaskActuationBounds> actuationTask_;
    tsid::solvers::SolverProxQP solver_;
    tsid::solvers::HQPOutput last_solution_;
    tsid::trajectories::TrajectorySample samplePosture_;
    tsid::trajectories::TrajectorySample sampleBase_;
  };

} // namespace simple_mpc
