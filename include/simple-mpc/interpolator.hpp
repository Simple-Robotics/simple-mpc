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

namespace simple_mpc
{
  class Interpolator
  {
  public:
    explicit Interpolator(const long nx, const long nv, const long nu, const long nf, const double MPC_timestep);

    void interpolate(
      const double delay,
      std::vector<Eigen::VectorXd> xs,
      std::vector<Eigen::VectorXd> us,
      std::vector<Eigen::VectorXd> ddqs,
      std::vector<Eigen::VectorXd> forces);

    Eigen::VectorXd x_interpolated_;
    Eigen::VectorXd u_interpolated_;
    Eigen::VectorXd a_interpolated_;
    Eigen::VectorXd forces_interpolated_;
    double MPC_timestep_;
  };
} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_INTERPOLATOR_HPP_
