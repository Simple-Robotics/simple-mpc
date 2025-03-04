///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "simple-mpc/fwd.hpp"
#include "simple-mpc/robot-handler.hpp"

#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/function-xpr-slice.hpp>
#include <proxsuite-nlp/modelling/constraints/box-constraint.hpp>
#include <proxsuite-nlp/modelling/constraints/negative-orthant.hpp>
#ifndef ALIGATOR_PINOCCHIO_V3
  #error "aligator was not compiled with Pinocchio 3 support. simple-mpc requires Pinocchio 3 features in aligator."
#endif

namespace simple_mpc
{
  using namespace aligator;
  using CostStack = CostStackTpl<double>;
  using ControlErrorResidual = ControlErrorResidualTpl<double>;
  using QuadraticControlCost = QuadraticControlCostTpl<double>;
  using QuadraticStateCost = QuadraticStateCostTpl<double>;
  using QuadraticResidualCost = QuadraticResidualCostTpl<double>;
  using StateErrorResidual = StateErrorResidualTpl<double>;
  using BoxConstraint = proxsuite::nlp::BoxConstraintTpl<double>;
  using NegativeOrthant = proxsuite::nlp::NegativeOrthantTpl<double>;
  using EqualityConstraint = proxsuite::nlp::EqualityConstraintTpl<double>;
  using FunctionSliceXpr = FunctionSliceXprTpl<double>;

#define SIMPLE_MPC_DEFINE_DEFAULT_MOVE_CTORS(Type)                                                                     \
  Type(Type &&) = default;                                                                                             \
  Type & operator=(Type &&) = default

  ///
  /// @brief Base abstract class for all MPC problems.
  ///
  class OCPHandler
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// Constructor
    explicit OCPHandler(const RobotModelHandler & model_handler);
    SIMPLE_MPC_DEFINE_DEFAULT_MOVE_CTORS(OCPHandler);
    virtual ~OCPHandler();

    /// Virtual functions defined in child classes

    // Create one instance of stage from desired contacts and forces
    virtual StageModel createStage(
      const std::map<std::string, bool> & contact_phase,
      const std::map<std::string, pinocchio::SE3> & contact_pose,
      const std::map<std::string, Eigen::VectorXd> & force_refs,
      const std::map<std::string, bool> & land_constraint) = 0;

    // Create the complete vector of stages from contact_sequence
    virtual std::vector<xyz::polymorphic<StageModel>> createStages(
      const std::vector<std::map<std::string, bool>> & contact_phases,
      const std::vector<std::map<std::string, pinocchio::SE3>> & contact_poses,
      const std::vector<std::map<std::string, Eigen::VectorXd>> & contact_forces);

    // Manage terminal cost and constraint
    virtual CostStack createTerminalCost() = 0;

    virtual void updateTerminalConstraint(const Eigen::Vector3d & com_ref) = 0;

    virtual void createTerminalConstraint(const Eigen::Vector3d & com_ref) = 0;

    // Setter and getter for poses reference
    virtual void
    setReferencePose(const std::size_t t, const std::string & ee_name, const pinocchio::SE3 & pose_ref) = 0;
    virtual void setReferencePoses(const std::size_t t, const std::map<std::string, pinocchio::SE3> & pose_refs) = 0;
    virtual void setTerminalReferencePose(const std::string & ee_name, const pinocchio::SE3 & pose_ref) = 0;
    virtual const pinocchio::SE3 getReferencePose(const std::size_t t, const std::string & ee_name) = 0;

    // Setter and getter for base velocity
    virtual const Eigen::VectorXd getVelocityBase(const std::size_t t) = 0;
    virtual void setVelocityBase(const std::size_t t, const ConstVectorRef & velocity_base) = 0;

    // Setter and getter for base pose
    virtual const Eigen::VectorXd getPoseBase(const std::size_t t) = 0;
    virtual void setPoseBase(const std::size_t t, const ConstVectorRef & pose_base) = 0;

    // Setter and getter for forces reference
    virtual void setReferenceForces(const std::size_t t, const std::map<std::string, Eigen::VectorXd> & force_refs) = 0;
    virtual void
    setReferenceForce(const std::size_t t, const std::string & ee_name, const ConstVectorRef & force_ref) = 0;
    virtual const Eigen::VectorXd getReferenceForce(const std::size_t t, const std::string & ee_name) = 0;
    virtual const Eigen::VectorXd getProblemState(const RobotDataHandler & data_handler) = 0;
    virtual size_t getContactSupport(const std::size_t t) = 0;
    virtual std::vector<bool> getContactState(const std::size_t t) = 0;
    virtual void setReferenceState(const std::size_t t, const ConstVectorRef & x_ref) = 0;
    virtual const ConstVectorRef getReferenceState(const std::size_t t) = 0;

    /// Common functions for all problems

    // Create one TrajOptProblem from contact sequence
    void createProblem(
      const ConstVectorRef & x0,
      const size_t horizon,
      const int force_size,
      const double gravity,
      const bool terminal_constraint);

    // Setter and getter for state and control reference
    void setReferenceControl(const std::size_t t, const ConstVectorRef & u_ref);
    ConstVectorRef getReferenceControl(const std::size_t t);

    // Getter for various objects and quantities
    CostStack * getCostStack(std::size_t t);
    CostStack * getTerminalCostStack();
    std::size_t getCostNumber() const;
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
      return nu_;
    }

  protected:
    // Size of the problem
    int nq_;
    int nv_;
    int ndx_;
    int nu_;
    bool problem_initialized_ = false;
    bool terminal_constraint_ = false;

    /// State reference
    Eigen::VectorXd x0_;

    /// The robot model
    RobotModelHandler model_handler_;

    /// The reference shooting problem storing all shooting nodes
    std::unique_ptr<TrajOptProblem> problem_;

    // Vector reference for control cost
    Eigen::VectorXd control_ref_;
  };

} // namespace simple_mpc
