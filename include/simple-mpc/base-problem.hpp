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

#include <aligator/core/cost-abstract.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/modelling/centroidal/centroidal-friction-cone.hpp>
#include <aligator/modelling/centroidal/centroidal-wrench-cone.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/dynamics/integrator-euler.hpp>
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <aligator/modelling/function-xpr-slice.hpp>
#include <aligator/modelling/multibody/center-of-mass-translation.hpp>
#include <aligator/modelling/multibody/centroidal-momentum.hpp>
#include <aligator/modelling/multibody/contact-force.hpp>
#include <aligator/modelling/multibody/dcm-position.hpp>
#include <aligator/modelling/multibody/frame-placement.hpp>
#include <aligator/modelling/multibody/frame-translation.hpp>
#include <aligator/modelling/multibody/frame-velocity.hpp>
#include <proxsuite-nlp/modelling/constraints/box-constraint.hpp>
#include <proxsuite-nlp/modelling/constraints/negative-orthant.hpp>
#ifndef ALIGATOR_PINOCCHIO_V3
#error "aligator no compile with pin v3"
#endif

namespace simple_mpc {
using namespace aligator;
using StageModel = StageModelTpl<double>;
using CostStack = CostStackTpl<double>;
using TrajOptProblem = TrajOptProblemTpl<double>;
using CostAbstract = CostAbstractTpl<double>;
using QuadraticControlCost = QuadraticControlCostTpl<double>;
using QuadraticStateCost = QuadraticStateCostTpl<double>;
using QuadraticResidualCost = QuadraticResidualCostTpl<double>;
using IntegratorSemiImplEuler = dynamics::IntegratorSemiImplEulerTpl<double>;
using IntegratorEuler = dynamics::IntegratorEulerTpl<double>;
using VectorSpace = proxsuite::nlp::VectorSpaceTpl<double>;
using ContactForceResidual = ContactForceResidualTpl<double>;
using DCMPositionResidual = DCMPositionResidualTpl<double>;
using FramePlacementResidual = FramePlacementResidualTpl<double>;
using FrameTranslationResidual = FrameTranslationResidualTpl<double>;
using CentroidalMomentumResidual = CentroidalMomentumResidualTpl<double>;
using FrameVelocityResidual = FrameVelocityResidualTpl<double>;
using StateErrorResidual = StateErrorResidualTpl<double>;
using CenterOfMassTranslationResidual =
    CenterOfMassTranslationResidualTpl<double>;
using BoxConstraint = proxsuite::nlp::BoxConstraintTpl<double>;
using NegativeOrthant = proxsuite::nlp::NegativeOrthantTpl<double>;
using EqualityConstraint = proxsuite::nlp::EqualityConstraintTpl<double>;
using FunctionSliceXpr = FunctionSliceXprTpl<double>;
using CentroidalWrenchConeResidual = CentroidalWrenchConeResidualTpl<double>;
using CentroidalFrictionConeResidual =
    CentroidalFrictionConeResidualTpl<double>;
/**
 * @brief Base problem abstract class
 */

class Problem {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Constructor
  Problem();
  Problem(const RobotDataHandler &handler);
  virtual ~Problem();

  /// Virtual functions defined in child classes

  // Create one instance of stage from desired contacts and forces
  virtual StageModel
  createStage(const std::map<std::string, bool> &contact_phase,
              const std::map<std::string, pinocchio::SE3> &contact_pose,
              const std::map<std::string, Eigen::VectorXd> &force_refs,
              const std::map<std::string, bool> &land_constraint) = 0;

  // Create the complete vector of stages from contact_sequence
  virtual std::vector<xyz::polymorphic<StageModel>> createStages(
      const std::vector<std::map<std::string, bool>> &contact_phases,
      const std::vector<std::map<std::string, pinocchio::SE3>> &contact_poses,
      const std::vector<std::map<std::string, Eigen::VectorXd>>
          &contact_forces);

  // Manage terminal cost and constraint
  virtual CostStack createTerminalCost() = 0;

  virtual void updateTerminalConstraint(const Eigen::Vector3d &com_ref) = 0;

  virtual void createTerminalConstraint() = 0;

  // Setter and getter for poses reference
  virtual void setReferencePose(const std::size_t t, const std::string &ee_name,
                                const pinocchio::SE3 &pose_ref) = 0;
  virtual void
  setReferencePoses(const std::size_t t,
                    const std::map<std::string, pinocchio::SE3> &pose_refs) = 0;
  virtual void setTerminalReferencePose(const std::string &ee_name,
                                        const pinocchio::SE3 &pose_ref) = 0;
  virtual const pinocchio::SE3 getReferencePose(const std::size_t t,
                                                const std::string &ee_name) = 0;

  // Setter and getter for base velocity
  virtual const Eigen::VectorXd getVelocityBase(const std::size_t t) = 0;
  virtual void setVelocityBase(const std::size_t t,
                               const Eigen::VectorXd &velocity_base) = 0;

  // Setter and getter for base pose
  virtual const Eigen::VectorXd getPoseBase(const std::size_t t) = 0;
  virtual void setPoseBase(const std::size_t t,
                           const Eigen::VectorXd &pose_base) = 0;

  // Setter and getter for forces reference
  virtual void setReferenceForces(
      const std::size_t t,
      const std::map<std::string, Eigen::VectorXd> &force_refs) = 0;
  virtual void setReferenceForce(const std::size_t t,
                                 const std::string &ee_name,
                                 const Eigen::VectorXd &force_ref) = 0;
  virtual const Eigen::VectorXd
  getReferenceForce(const std::size_t t, const std::string &ee_name) = 0;
  virtual const Eigen::VectorXd getProblemState() = 0;
  virtual size_t getContactSupport(const std::size_t t) = 0;

  /// Common functions for all problems

  // Create one TrajOptProblem from contact sequence
  void createProblem(const Eigen::VectorXd &x0, const size_t horizon,
                     const int force_size, const double gravity);

  // Setter and getter for control reference
  void setReferenceControl(const std::size_t t, const Eigen::VectorXd &u_ref);
  const Eigen::VectorXd getReferenceControl(const std::size_t t);

  // Getter for various objects and quantities
  CostStack *getCostStack(std::size_t t);
  CostStack *getTerminalCostStack();
  std::size_t getCostNumber();
  std::size_t getSize();
  std::shared_ptr<TrajOptProblem> getProblem() { return problem_; }
  const RobotDataHandler &getDataHandler() const { return robot_data_handler_; }
  const RobotModelHandler &getModelHandler() const { return robot_model_handler_; }
  int getNu() { return nu_; }

protected:
  // Size of the problem
  int nq_;
  int nv_;
  int ndx_;
  int nu_;
  bool problem_initialized_ = false;
  bool terminal_constraint_ = false;

  /// The robot model
  RobotDataHandler robot_data_handler_;
  RobotModelHandler robot_model_handler_;

  /// The reference shooting problem storing all shooting nodes
  std::shared_ptr<TrajOptProblem> problem_;

  // Vector reference for control cost
  Eigen::VectorXd control_ref_;
};

} // namespace simple_mpc
