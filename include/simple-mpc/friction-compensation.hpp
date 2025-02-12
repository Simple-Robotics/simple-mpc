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
     * @param model Pinocchio model containing friction coefficients
       @param with_free_flyer Bool indicating if model has free flyer joint
     */
    FrictionCompensation(const Model & model, const bool with_free_flyer = true);

    /**
     * @brief Add friction correction to joint torque
     *
     * @param[in] velocity Joint velocity
     * @param[in] torque Joint torque
     */
    void computeFriction(Eigen::Ref<const VectorXd> velocity, Eigen::Ref<VectorXd> torque);

    // Sign function for internal computation
    static double signFunction(double x);

    // Actuation size
    int nu_;

    // Friction coefficients
    Eigen::VectorXd dry_friction_;
    Eigen::VectorXd viscuous_friction_;
  };

} // namespace simple_mpc
