///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2025, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <pinocchio/fwd.hpp>
// Include pinocchio first
#include <Eigen/Dense>

#include "simple-mpc/fwd.hpp"
#include <pinocchio/multibody/model.hpp>

namespace simple_mpc
{
  using namespace pinocchio;

  /**
   * @brief Class managing the friction compensation for torque
   *
   * It applies a compensation term to a torque depending on dry and
   * viscuous frictions.
   */
  class FrictionCompensation
  {
  public:
    /**
     * @brief Construct a new Friction Compensation object
     *
     * @param actuation_size Dimension of torque
     */
    FrictionCompensation(const Model & model, const long actuation_size);

    /**
     * @brief Compute the torque correction due to friction and store it internally.
     *
     * @param[in] velocity Joint velocity
     * @param[in] torque Joint torque
     */
    void computeFriction(const Eigen::VectorXd & velocity, const Eigen::VectorXd & torque);

    // Sign function for internal computation
    static double signFunction(double x);

    // Internal torque with friction compensation
    Eigen::VectorXd corrected_torque_;

    // Friction coefficients
    Eigen::VectorXd dry_friction_;
    Eigen::VectorXd viscuous_friction_;
  };

} // namespace simple_mpc
