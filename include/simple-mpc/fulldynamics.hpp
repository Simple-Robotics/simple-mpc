///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <pinocchio/algorithm/proximal.hpp>

#include "simple-mpc/ocp-handler.hpp"

namespace simple_mpc
{
  using namespace aligator;
  using Base = OCPHandler;
  using ProximalSettings = pinocchio::ProximalSettingsTpl<double>;

  /**
   * @brief Build a full dynamics problem based on the
   * MultibodyConstraintFwdDynamics of Aligator.
   *
   * State is defined as concatenation of joint positions and
   * joint velocities; control is defined as joint torques.
   */

  struct FullDynamicsSettings
  {
  public:
    // timestep in problem shooting nodes
    double timestep;

    // Cost function weights
    Eigen::MatrixXd w_x;      // State
    Eigen::MatrixXd w_u;      // Control
    Eigen::MatrixXd w_cent;   // Centroidal momentum
    Eigen::MatrixXd w_forces; // Contact forces
    Eigen::MatrixXd w_frame;  // End effector placement

    // Physics parameters
    Eigen::Vector3d gravity;
    int force_size;
    double mu;
    double Lfoot; // Half-length of foot (if contact 6D)
    double Wfoot; // Half-width of foot (if contact 6D)

    // Constraints
    bool torque_limits;
    bool kinematics_limits;
    bool force_cone;
    bool land_cstr;

    // Control limits
    Eigen::VectorXd umin;
    Eigen::VectorXd umax;

    // Kinematics limits
    Eigen::VectorXd qmin;
    Eigen::VectorXd qmax;

    // Baumgarte gains
    Eigen::VectorXd Kp_correction;
    Eigen::VectorXd Kd_correction;
  };

  class FullDynamicsOCP : public OCPHandler
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructors
    explicit FullDynamicsOCP(const FullDynamicsSettings & settings, const RobotModelHandler & model_handler);

    SIMPLE_MPC_DEFINE_DEFAULT_MOVE_CTORS(FullDynamicsOCP);

    virtual ~FullDynamicsOCP() = default;

    // Create one FullDynamics stage
    StageModel createStage(
      const std::map<std::string, bool> & contact_phase,
      const std::map<std::string, pinocchio::SE3> & contact_pose,
      const std::map<std::string, Eigen::VectorXd> & contact_force,
      const std::map<std::string, bool> & land_constraints) override;

    // Manage terminal cost and constraint
    CostStack createTerminalCost() override;
    void createTerminalConstraint(const Eigen::Vector3d & com_ref) override;
    void updateTerminalConstraint(const Eigen::Vector3d & com_ref) override;

    // Getters and setters
    void setReferencePose(const std::size_t t, const std::string & ee_name, const pinocchio::SE3 & pose_ref) override;
    void setReferencePoses(const std::size_t t, const std::map<std::string, pinocchio::SE3> & pose_refs) override;
    void setTerminalReferencePose(const std::string & ee_name, const pinocchio::SE3 & pose_ref) override;
    void setReferenceForces(const std::size_t t, const std::map<std::string, Eigen::VectorXd> & force_refs) override;
    void setReferenceForce(const std::size_t t, const std::string & ee_name, const ConstVectorRef & force_ref) override;
    const pinocchio::SE3 getReferencePose(const std::size_t t, const std::string & cost_name) override;
    const Eigen::VectorXd getReferenceForce(const std::size_t t, const std::string & cost_name) override;
    const Eigen::VectorXd getVelocityBase(const std::size_t t) override;
    void setVelocityBase(const std::size_t t, const ConstVectorRef & velocity_base) override;
    const Eigen::VectorXd getPoseBase(const std::size_t t) override;
    void setPoseBase(const std::size_t t, const ConstVectorRef & pose_base) override;
    const Eigen::VectorXd getProblemState(const RobotDataHandler & data_handler) override;
    size_t getContactSupport(const std::size_t t) override;
    std::vector<bool> getContactState(const std::size_t t) override;
    void setReferenceState(const std::size_t t, const ConstVectorRef & x_ref) override;
    const ConstVectorRef getReferenceState(const std::size_t t) override;
    FullDynamicsSettings getSettings()
    {
      return settings_;
    }

  protected:
    // Problem settings
    FullDynamicsSettings settings_;
    ProximalSettings prox_settings_;

    // Actuation matrix
    Eigen::MatrixXd actuation_matrix_;

    // Complete list of contact models to compute dynamics
    pinocchio::context::RigidConstraintModelVector constraint_models_;
  };

} // namespace simple_mpc
