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

  Interpolator::Interpolator(const Model & model)
  {
    model_ = model;
  }

  void Interpolator::interpolateConfiguration(
    const double delay,
    const double timestep,
    const std::vector<Eigen::VectorXd> & qs,
    Eigen::Ref<Eigen::VectorXd> q_interp)
  {
    assert(("Configuration is not of the right size", qs[0].size() == model_.nq));

    // Compute the time knot corresponding to the current delay
    size_t step_nb = static_cast<size_t>(delay / timestep);
    double step_progress = (delay - (double)step_nb * timestep) / timestep;

    // Interpolate configuration trajectory
    if (step_nb >= qs.size() - 1)
      q_interp = qs.back();
    else
    {
      q_interp = pinocchio::interpolate(model_, qs[step_nb], qs[step_nb + 1], step_progress);
    }
  }

  void Interpolator::interpolateLinear(
    const double delay,
    const double timestep,
    const std::vector<Eigen::VectorXd> & vs,
    Eigen::Ref<Eigen::VectorXd> v_interp)
  {
    // Compute the time knot corresponding to the current delay
    size_t step_nb = static_cast<size_t>(delay / timestep);
    double step_progress = (delay - (double)step_nb * timestep) / timestep;

    // Interpolate configuration trajectory
    if (step_nb >= vs.size() - 1)
      v_interp = vs.back();
    else
    {
      v_interp = vs[step_nb + 1] * step_progress + vs[step_nb] * (1. - step_progress);
    }
  }

} // namespace simple_mpc
