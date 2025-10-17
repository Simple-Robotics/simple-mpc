///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "aligator/modelling/dynamics/multibody-free-fwd.hpp"
#include "simple-mpc/arm-dynamics.hpp"
#include <aligator/core/stage-data.hpp>
#include <aligator/fwd.hpp>
#include <aligator/modelling/dynamics/fwd.hpp>
#include <aligator/modelling/dynamics/integrator-explicit.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>

#include "simple-mpc/deprecated.hpp"
#include "simple-mpc/fwd.hpp"
#include "simple-mpc/robot-handler.hpp"

namespace simple_mpc
{
  using ExplicitIntegratorData = dynamics::ExplicitIntegratorDataTpl<double>;
  using MultibodyFreeFwdDataT = dynamics::MultibodyFreeFwdDataTpl<double>;
  using MultibodyFreeFwdDynamicsT = dynamics::MultibodyFreeFwdDynamicsTpl<double>;

  struct ArmMPCSettings
  {
  public:
    // Solver-related quantities
    double TOL = 1e-4;
    double mu_init = 1e-8;
    std::size_t max_iters = 1;
    std::size_t num_threads = 2;

    // Timings
    double timestep = 0.01;
  };

  /**
   * @brief Build a MPC object holding an instance
   * of a trajectory optimization problem
   */
  class ArmMPC
  {

  protected:
    enum MoveType
    {
      RESTING,
      REACHING
    };

    // INTERNAL UPDATING function
    void updateTargetReference();

    // Memory preallocations:
    std::vector<unsigned long> controlled_joints_id_;
    std::string ee_name_;
    Eigen::VectorXd x_internal_;
    bool time_to_solve_ddp_ = false;

    std::shared_ptr<RobotDataHandler> data_handler_;

  public:
    std::unique_ptr<SolverProxDDP> solver_;
    ArmMPCSettings settings_;
    std::shared_ptr<ArmDynamicsOCP> ocp_handler_;

    explicit ArmMPC(const ArmMPCSettings & settings, std::shared_ptr<ArmDynamicsOCP> problem);

    // Generate the cycle walking problem along which we will iterate
    // the receding horizon
    void generateReachHorizon(const Eigen::Vector3d & reach_pose);

    // Perform one iteration of MPC
    void iterate(const ConstVectorRef & x);

    // Recede the horizon
    void recedeWithCycle();

    // Getters and setters
    void setReferencePose(const std::size_t t, const Eigen::Vector3d & pose_ref);

    const Eigen::Vector3d getReferencePose(const std::size_t t) const;

    void setReferenceState(const VectorXd & state_ref)
    {
      x_reference_ = state_ref;
    }

    // getters and setters
    TrajOptProblem & getTrajOptProblem();

    const RobotDataHandler & getDataHandler() const
    {
      return *data_handler_;
    }
    const RobotModelHandler & getModelHandler() const
    {
      return ocp_handler_->getModelHandler();
    }

    const ConstVectorRef getStateDerivative(const std::size_t t);

    void switchToReach(const Eigen::Vector3d & reach_pose);

    void switchToRest();

    // Solution vectors for state and control
    std::vector<VectorXd> xs_;
    std::vector<VectorXd> us_;
    // Riccati gains
    std::vector<MatrixXd> Ks_;
    VectorXd x_reference_;
    MoveType now_;
    Eigen::Vector3d reach_pose_;

    // Initial quantities
    VectorXd x0_;
    VectorXd u0_;
  };

} // namespace simple_mpc
