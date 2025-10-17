///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <aligator/context.hpp>
#include <aligator/modelling/contact-map.hpp>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace simple_mpc
{
  namespace pin = pinocchio;
  using pin::FrameIndex;
  using std::shared_ptr;
  using ContactMap = aligator::ContactMapTpl<double>;

  /// ALIGATOR TYPEDEFS / USING-DECLS

  using aligator::context::SolverProxDDP;
  using aligator::context::StageData;
  using aligator::context::StageModel;
  using aligator::context::TrajOptProblem;

  /// SIMPLE-MPC FORWARD DECLARATIONS

  struct Settings;
  struct FullDynamicsSettings;
  class RobotDataHandler;
  class FullDynamicsOCP;
  class KinodynamicsOCP;
  class CentroidalOCP;
  class ArmDynamicsOCP;
  class OCPHandler;
  class IDSolver;
  class IKIDSolver;
  class FrictionCompensation;

  /// EIGEN TYPEDEFS

  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  using MatrixRef = Eigen::Ref<MatrixXd>;
  using VectorRef = Eigen::Ref<VectorXd>;
  using ConstMatrixRef = Eigen::Ref<const MatrixXd>;
  using ConstVectorRef = Eigen::Ref<const VectorXd>;

  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Vector7d = Eigen::Matrix<double, 7, 1>;

} // namespace simple_mpc
