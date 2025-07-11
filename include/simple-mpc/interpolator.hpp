///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2025, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
/**
 * @file interpolator.hpp
 * @brief Interpolation class for practical control of the robot
 */

#pragma once

#include <pinocchio/algorithm/joint-configuration.hpp>

#include "simple-mpc/fwd.hpp"
#include "simple-mpc/model-utils.hpp"

namespace simple_mpc
{
  class Interpolator
  {
  public:
    explicit Interpolator(const Model & model)
    : model_(model) {};

    void interpolateConfiguration(
      const double delay,
      const double timestep,
      const std::vector<Eigen::VectorXd> & qs,
      Eigen::Ref<Eigen::VectorXd> q_interp);

    void interpolateState(
      const double delay,
      const double timestep,
      const std::vector<Eigen::VectorXd> & xs,
      Eigen::Ref<Eigen::VectorXd> x_interp);

    void interpolateLinear(
      const double delay,
      const double timestep,
      const std::vector<Eigen::VectorXd> & vs,
      Eigen::Ref<Eigen::VectorXd> v_interp);

    void interpolateContacts(
      const double delay,
      const double timestep,
      const std::vector<std::vector<bool>> & cs,
      std::vector<bool> & c_interp);

    // Pinocchio model
    Model model_;
  };

} // namespace simple_mpc
