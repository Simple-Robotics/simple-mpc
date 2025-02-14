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

  StateInterpolator::StateInterpolator(const Model & model)
  {
    model_ = model;
    diff_q_.resize(model_.nv);
  }

  void StateInterpolator::interpolate(
    const double delay,
    const double timestep,
    const std::vector<Eigen::VectorXd> xs,
    Eigen::Ref<Eigen::VectorXd> x_interp)
  {
    // Compute the time knot corresponding to the current delay
    size_t step_nb = static_cast<size_t>(delay / timestep);
    double step_progress = (delay - (double)step_nb * timestep) / timestep;

    // Interpolate state and command trajectories
    if (step_nb >= xs.size() - 1)
      x_interp = xs.back();
    else
    {
      // Compute the differential between configuration
      diff_q_ = pinocchio::difference(model_, xs[step_nb].head(model_.nq), xs[step_nb + 1].head(model_.nq));

      pinocchio::integrate(model_, xs[step_nb].head(model_.nq), diff_q_ * step_progress, x_interp.head(model_.nq));

      // Compute velocity interpolation
      x_interp.tail(model_.nv) =
        xs[step_nb + 1].tail(model_.nv) * step_progress + xs[step_nb].tail(model_.nv) * (1. - step_progress);
    }
  }

  LinearInterpolator::LinearInterpolator(const size_t vec_size)
  {
    vec_size_ = vec_size;
  }

  void LinearInterpolator::interpolate(
    const double delay,
    const double timestep,
    const std::vector<Eigen::VectorXd> vecs,
    Eigen::Ref<Eigen::VectorXd> vec_interp)
  {
    // Compute the time knot corresponding to the current delay
    size_t step_nb = static_cast<size_t>(delay / timestep);
    double step_progress = (delay - (double)step_nb * timestep) / timestep;

    // Interpolate state and command trajectories
    if (step_nb >= vecs.size() - 1)
      vec_interp = vecs.back();
    else
    {
      vec_interp = vecs[step_nb + 1] * step_progress + vecs[step_nb] * (1. - step_progress);
    }
  }

} // namespace simple_mpc
