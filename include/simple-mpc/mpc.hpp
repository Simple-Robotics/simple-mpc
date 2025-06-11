///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <aligator/core/stage-data.hpp>
#include <aligator/fwd.hpp>
#include <aligator/modelling/dynamics/fwd.hpp>
#include <aligator/modelling/dynamics/integrator-explicit.hpp>
#include <aligator/modelling/dynamics/multibody-constraint-fwd.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>

#include "simple-mpc/deprecated.hpp"
#include "simple-mpc/foot-trajectory.hpp"
#include "simple-mpc/fwd.hpp"
#include "simple-mpc/ocp-handler.hpp"
#include "simple-mpc/robot-handler.hpp"

namespace simple_mpc
{
  using ExplicitIntegratorData = dynamics::ExplicitIntegratorDataTpl<double>;
  using MultibodyConstraintFwdData = dynamics::MultibodyConstraintFwdDataTpl<double>;
  using MultibodyConstraintFwdDynamics = dynamics::MultibodyConstraintFwdDynamicsTpl<double>;

  struct MPCSettings
  {
  public:
    // Step-related quantities
    double swing_apex = 0.15;

    // Force parameters
    double support_force = 1000;

    // Solver-related quantities
    double TOL = 1e-4;
    double mu_init = 1e-8;
    std::size_t max_iters = 1;
    std::size_t num_threads = 2;

    // Timings
    int T_fly = 80;
    int T_contact = 20;
    size_t T = 100;
    double timestep = 0.01;
  };

  /**
   * @brief Build a MPC object holding an instance
   * of a trajectory optimization problem
   */
  class MPC
  {

  protected:
    enum LocomotionType
    {
      WALKING,
      STANDING,
      MOTION
    };

    std::vector<std::map<std::string, bool>> contact_states_;
    std::vector<std::shared_ptr<StageModel>> cycle_horizon_;
    std::vector<std::shared_ptr<StageData>> cycle_horizon_data_;
    std::vector<std::shared_ptr<StageModel>> one_horizon_;
    std::vector<std::shared_ptr<StageData>> one_horizon_data_;
    std::vector<std::map<std::string, bool>> one_contact_states_;
    std::vector<std::shared_ptr<StageModel>> standing_horizon_;
    std::vector<std::shared_ptr<StageData>> standing_horizon_data_;
    FootTrajectory foot_trajectories_;
    std::map<std::string, pinocchio::SE3> relative_feet_poses_;
    // INTERNAL UPDATING function
    void updateStepTrackerReferences();

    // Memory preallocations:
    std::vector<unsigned long> controlled_joints_id_;
    std::vector<std::string> ee_names_;
    Eigen::VectorXd x_internal_;
    bool time_to_solve_ddp_ = false;
    Eigen::Vector3d com0_;
    LocomotionType now_;

    std::shared_ptr<RobotDataHandler> data_handler_;

  public:
    std::unique_ptr<SolverProxDDP> solver_;
    Vector6d velocity_base_;
    Eigen::Vector3d next_pose_;
    Eigen::Vector2d twist_vect_;
    MPCSettings settings_;
    std::shared_ptr<OCPHandler> ocp_handler_;

    explicit MPC(const MPCSettings & settings, std::shared_ptr<OCPHandler> problem);

    // Generate the cycle walking problem along which we will iterate
    // the receding horizon
    void generateCycleHorizon(const std::vector<std::map<std::string, bool>> & contact_states);

    // Generate a motion horizon problem for unique motion
    void generateOneHorizon(const std::vector<std::map<std::string, bool>> & contact_states);

    // Perform one iteration of MPC
    void iterate(const ConstVectorRef & x);

    void updateCycleTiming(const bool updateOnlyHorizon);

    // Recede the horizon
    void recedeWithCycle();

    // Getters and setters
    void setReferencePose(const std::size_t t, const std::string & ee_name, const pinocchio::SE3 & pose_ref);

    void setTerminalReferencePose(const std::string & ee_name, const pinocchio::SE3 & pose_ref);

    const pinocchio::SE3 getReferencePose(const std::size_t t, const std::string & ee_name) const;

    SIMPLE_MPC_DEPRECATED void setVelocityBase(const Vector6d & v)
    {
      velocity_base_ = v;
    }

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

    std::vector<std::shared_ptr<StageModel>> & getCycleHorizon()
    {
      return cycle_horizon_;
    }

    inline bool getCyclingContactState(const std::size_t t, const std::string & ee_name) const
    {
      return contact_states_[t].at(ee_name);
    }

    inline int getFootTakeoffCycle(const std::string & ee_name) const
    {
      if (foot_takeoff_times_.at(ee_name).empty())
      {
        return -1;
      }
      else
      {
        return foot_takeoff_times_.at(ee_name)[0];
      }
    }
    inline int getFootLandCycle(const std::string & ee_name) const
    {
      if (foot_land_times_.at(ee_name).empty())
      {
        return -1;
      }
      else
      {
        return foot_land_times_.at(ee_name)[0];
      }
    }

    const ConstVectorRef getStateDerivative(const std::size_t t);

    /**
     * @brief Return contact forces for a full dynamics MPC problem
     * @warning Only work with fulldynamics OCP handler
     */
    const Eigen::VectorXd getContactForces(const std::size_t t);

    void switchToWalk(const Vector6d & velocity_base);

    void switchToStand();

    // Footstep timings for each end effector
    std::map<std::string, std::vector<int>> foot_takeoff_times_, foot_land_times_;
    std::size_t one_horizon_iterator_;

    // Solution vectors for state and control
    std::vector<VectorXd> xs_;
    std::vector<VectorXd> us_;
    // Riccati gains
    std::vector<MatrixXd> Ks_;
    VectorXd x_reference_;

    // Initial quantities
    VectorXd x0_;
    VectorXd u0_;
  };

} // namespace simple_mpc
