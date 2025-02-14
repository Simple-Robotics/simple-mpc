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

#ifndef SIMPLE_MPC_INTERPOLATOR_HPP_
#define SIMPLE_MPC_INTERPOLATOR_HPP_

#include "simple-mpc/fwd.hpp"
#include "simple-mpc/model-utils.hpp"

namespace simple_mpc
{
  class StateInterpolator
  {
  public:
    explicit StateInterpolator(const Model & model);

    void interpolate(
      const double delay,
      const double timestep,
      const std::vector<Eigen::VectorXd> xs,
      Eigen::Ref<Eigen::VectorXd> x_interp);

    // Intermediate differential configuration
    Eigen::VectorXd diff_q_;

    // Pinocchio model
    Model model_;
  };

  class LinearInterpolator
  {
  public:
    explicit LinearInterpolator(const size_t vec_size);

    void interpolate(
      const double delay,
      const double timestep,
      const std::vector<Eigen::VectorXd> vecs,
      Eigen::Ref<Eigen::VectorXd> vec_interp);

    size_t vec_size_;
  };
} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_INTERPOLATOR_HPP_
