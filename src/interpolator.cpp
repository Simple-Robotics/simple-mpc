///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2025, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "simple-mpc/interpolator.hpp"

namespace simple_mpc
{

  Interpolator::Interpolator(const long nx, const long nv, const long nu, const long nf, const double MPC_timestep)
  : MPC_timestep_(MPC_timestep)
  {
    x_interpolated_.resize(nx);
    u_interpolated_.resize(nu);
    a_interpolated_.resize(nv);
    forces_interpolated_.resize(nf);
  }

  void Interpolator::interpolate(
    const double delay,
    std::vector<Eigen::VectorXd> xs,
    std::vector<Eigen::VectorXd> us,
    std::vector<Eigen::VectorXd> ddqs,
    std::vector<Eigen::VectorXd> forces)
  {
    // Compute the time knot corresponding to the current delay
    size_t step_nb = static_cast<size_t>(delay / MPC_timestep_);
    double step_progress = (delay - (double)step_nb * MPC_timestep_) / MPC_timestep_;

    // Interpolate state and command trajectories
    if (step_nb >= xs.size() - 1)
    {
      step_nb = xs.size() - 1;
      step_progress = 0.0;
      x_interpolated_ = xs[step_nb];
      u_interpolated_ = us[step_nb];
      a_interpolated_ = ddqs[step_nb];
      forces_interpolated_ = forces[step_nb];
    }
    else
    {
      x_interpolated_ = xs[step_nb + 1] * step_progress + xs[step_nb] * (1. - step_progress);
      u_interpolated_ = us[step_nb + 1] * step_progress + us[step_nb] * (1. - step_progress);
      a_interpolated_ = ddqs[step_nb + 1] * step_progress + ddqs[step_nb] * (1. - step_progress);
      forces_interpolated_ = forces[step_nb + 1] * step_progress + forces[step_nb] * (1. - step_progress);
    }
  }

} // namespace simple_mpc
