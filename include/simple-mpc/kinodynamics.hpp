///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "simple-mpc/ocp-handler.hpp"

namespace simple_mpc
{
  using namespace aligator;
  /**
   * @brief Build a kinodynamics problem based on
   * aligator's KinodynamicsFwdDynamics class.
   *
   * State is defined as concatenation of joint positions and
   * joint velocities; control is defined as concatenation of
   * contact forces and joint acceleration.
   */

  struct KinodynamicsSettings
  {
    /// timestep in problem shooting nodes
    double timestep;

    // Cost function weights
    Eigen::MatrixXd w_x;       // State
    Eigen::MatrixXd w_u;       // Control
    Eigen::MatrixXd w_frame;   // End effector placement
    Eigen::MatrixXd w_cent;    // Centroidal momentum
    Eigen::MatrixXd w_centder; // Derivative of centroidal momentum

    // Kinematics limits
    Eigen::VectorXd qmin;
    Eigen::VectorXd qmax;

    // Physics parameters
    Eigen::Vector3d gravity;
    double mu;
    double Lfoot;
    double Wfoot;
    int force_size;

    // Constraint
    bool kinematics_limits;
    bool force_cone;
  };

  class KinodynamicsOCP : public OCPHandler
  {
    using Base = OCPHandler;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    explicit KinodynamicsOCP(const KinodynamicsSettings & settings, const RobotModelHandler & model_handler);

    SIMPLE_MPC_DEFINE_DEFAULT_MOVE_CTORS(KinodynamicsOCP);

    virtual ~KinodynamicsOCP() {};

    // Create one Kinodynamics stage
    StageModel createStage(
      const std::map<std::string, bool> & contact_phase,
      const std::map<std::string, pinocchio::SE3> & contact_pose,
      const std::map<std::string, Eigen::VectorXd> & contact_force,
      const std::map<std::string, bool> & land_constraint) override;

    // Manage terminal cost and constraint
    CostStack createTerminalCost() override;
    void createTerminalConstraint(const Eigen::Vector3d & com_ref) override;
    void updateTerminalConstraint(const Eigen::Vector3d & com_ref) override;

    // Getters and setters
    void setReferencePose(const std::size_t t, const std::string & ee_name, const pinocchio::SE3 & pose_ref) override;
    void setReferencePoses(const std::size_t i, const std::map<std::string, pinocchio::SE3> & pose_refs) override;
    void setTerminalReferencePose(const std::string & ee_name, const pinocchio::SE3 & pose_ref) override;
    void setReferenceForces(const std::size_t i, const std::map<std::string, Eigen::VectorXd> & force_refs) override;
    void setReferenceForce(const std::size_t i, const std::string & ee_name, const ConstVectorRef & force_ref) override;
    const Eigen::VectorXd getReferenceForce(const std::size_t i, const std::string & cost_name) override;
    const pinocchio::SE3 getReferencePose(const std::size_t i, const std::string & cost_name) override;
    const Eigen::VectorXd getVelocityBase(const std::size_t t) override;
    const Eigen::VectorXd getPoseBase(const std::size_t t) override;
    void setPoseBase(const std::size_t t, const ConstVectorRef & pose_base) override;
    void setVelocityBase(const std::size_t t, const ConstVectorRef & velocity_base) override;
    const Eigen::VectorXd getProblemState(const RobotDataHandler & data_handler) override;
    size_t getContactSupport(const std::size_t t) override;
    std::vector<bool> getContactState(const std::size_t t) override;
    void setReferenceState(const std::size_t t, const ConstVectorRef & x_ref) override;
    const ConstVectorRef getReferenceState(const std::size_t t) override;

    void computeControlFromForces(const std::map<std::string, Eigen::VectorXd> & force_refs);

    KinodynamicsSettings getSettings()
    {
      return settings_;
    }

  protected:
    KinodynamicsSettings settings_;
  };

} // namespace simple_mpc
