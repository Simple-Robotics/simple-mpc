///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2025, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "aligator/modelling/dynamics/multibody-constraint-fwd.hpp"
#include <pinocchio/algorithm/proximal.hpp>
#include <pinocchio/multibody/frame.hpp>

#include "simple-mpc/ocp-handler.hpp"

namespace simple_mpc
{
  using namespace aligator;
  using MultibodyConstraintFwdDynamics = dynamics::MultibodyConstraintFwdDynamicsTpl<double>;

  /**
   * @brief Build an arm dynamics problem based on the
   * MultibodyFreeFwdDynamics of Aligator.
   *
   * State is defined as concatenation of joint positions and
   * joint velocities; control is defined as joint torques.
   */

  struct ArmDynamicsSettings
  {
  public:
    // timestep in problem shooting nodes
    double timestep;

    // Cost function weights
    Eigen::MatrixXd w_x;      // State
    Eigen::MatrixXd w_u;      // Control
    Eigen::MatrixXd w_frame;  // End effector placement
    Eigen::MatrixXd w_forces; // Contact force

    // Physics parameters
    Eigen::Vector3d gravity;

    // Constraints
    bool torque_limits;
    bool kinematics_limits;

    // Control limits
    Eigen::VectorXd umin;
    Eigen::VectorXd umax;

    // Kinematics limits
    Eigen::VectorXd qmin;
    Eigen::VectorXd qmax;

    // Baumgarte gains
    Eigen::VectorXd Kp_correction;
    Eigen::VectorXd Kd_correction;

    std::string ee_name;
  };

  class ArmDynamicsOCP
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructors
    explicit ArmDynamicsOCP(const ArmDynamicsSettings & settings, const RobotModelHandler & model_handler);

    SIMPLE_MPC_DEFINE_DEFAULT_MOVE_CTORS(ArmDynamicsOCP);

    virtual ~ArmDynamicsOCP() = default;

    // Create one TrajOptProblem
    void createProblem(const ConstVectorRef & x0, const size_t horizon);

    // Create one ArmDynamics stage
    StageModel createStage(
      const bool reaching = false,
      const pinocchio::SE3 & reach_pose = pinocchio::SE3::Identity(),
      const bool is_contact = false,
      const Eigen::Vector3d & contact_force = Eigen::Vector3d::Zero());

    // Manage terminal cost and constraint
    CostStack createTerminalCost();

    // Getters and setters
    MultibodyConstraintFwdDynamics * getDynamics(std::size_t t);
    CostStack * getCostStack(std::size_t t);
    CostStack * getTerminalCostStack();
    void deactivateReach(const std::size_t t);
    void activateReach(const std::size_t t);
    void setReferencePose(const std::size_t t, const pinocchio::SE3 & pose_ref);
    const pinocchio::SE3 getReferencePose(const std::size_t t);
    void setTerminalReferencePose(const pinocchio::SE3 & pose_ref);
    const pinocchio::SE3 getTerminalReferencePose();
    void setReferenceForce(const std::size_t t, const Eigen::Vector3d & force_ref);
    const Eigen::Vector3d getReferenceForce(const std::size_t t);
    const Eigen::VectorXd getProblemState(const RobotDataHandler & data_handler);
    void setReferenceState(const std::size_t t, const ConstVectorRef & x_ref);
    const ConstVectorRef getReferenceState(const std::size_t t);
    void setWeight(const std::size_t t, const std::string key, double weight);
    double getWeight(const std::size_t t, const std::string key);
    void setTerminalWeight(const std::string key, double weight);
    double getTerminalWeight(const std::string key);
    void removeContact(const std::size_t t);
    void addContact(const std::size_t t);

    ArmDynamicsSettings getSettings()
    {
      return settings_;
    }
    std::size_t getSize() const
    {
      return problem_->numSteps();
    }
    TrajOptProblem & getProblem()
    {
      assert(problem_);
      return *problem_;
    }

    const TrajOptProblem & getProblem() const
    {
      assert(problem_);
      return *problem_;
    }

    const RobotModelHandler & getModelHandler() const
    {
      return model_handler_;
    }
    int getNu()
    {
      return nv_;
    }

  protected:
    // Problem settings
    ArmDynamicsSettings settings_;
    pinocchio::FrameIndex ee_id_;
    ProximalSettings prox_settings_;
    Eigen::MatrixXd actuation_matrix_;

    // Size of the problem
    int nq_;
    int nv_;
    int ndx_;
    bool problem_initialized_ = false;
    bool terminal_constraint_ = false;

    /// State reference
    Eigen::VectorXd x0_;

    /// The robot model
    RobotModelHandler model_handler_;

    // Complete list of contact models to compute dynamics
    pinocchio::context::RigidConstraintModelVector constraint_models_;

    /// The reference shooting problem storing all shooting nodes
    std::unique_ptr<TrajOptProblem> problem_;
  };

} // namespace simple_mpc
