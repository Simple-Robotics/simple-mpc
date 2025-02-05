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
    step_nb_ = static_cast<size_t>(delay / MPC_timestep_);
    step_progress_ = (delay - (double)step_nb_ * MPC_timestep_) / MPC_timestep_;

    // Interpolate state and command trajectories
    if (step_nb_ >= xs.size() - 1)
    {
      step_nb_ = xs.size() - 1;
      step_progress_ = 0.0;
      x_interpolated_ = xs[step_nb_];
      u_interpolated_ = us[step_nb_];
      a_interpolated_ = ddqs[step_nb_];
      forces_interpolated_ = forces[step_nb_];
    }
    else
    {
      x_interpolated_ = xs[step_nb_ + 1] * step_progress_ + xs[step_nb_] * (1. - step_progress_);
      u_interpolated_ = us[step_nb_ + 1] * step_progress_ + us[step_nb_] * (1. - step_progress_);
      a_interpolated_ = ddqs[step_nb_ + 1] * step_progress_ + ddqs[step_nb_] * (1. - step_progress_);
      forces_interpolated_ = forces[step_nb_ + 1] * step_progress_ + forces[step_nb_] * (1. - step_progress_);
    }
  }

} // namespace simple_mpc
