#include "simple-mpc/interpolator.hpp"

namespace simple_mpc
{
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

  void Interpolator::interpolateState(
    const double delay,
    const double timestep,
    const std::vector<Eigen::VectorXd> & xs,
    Eigen::Ref<Eigen::VectorXd> x_interp)
  {
    assert(("State is not of the right size", xs[0].size() == model_.nq + model_.nv));

    // Compute the time knot corresponding to the current delay
    size_t step_nb = static_cast<size_t>(delay / timestep);
    double step_progress = (delay - (double)step_nb * timestep) / timestep;

    // Interpolate state trajectory
    if (step_nb >= xs.size() - 1)
      x_interp = xs.back();
    else
    {
      x_interp.head(model_.nq) =
        pinocchio::interpolate(model_, xs[step_nb].head(model_.nq), xs[step_nb + 1].head(model_.nq), step_progress);
      x_interp.tail(model_.nv) =
        xs[step_nb + 1].tail(model_.nv) * step_progress + xs[step_nb].tail(model_.nv) * (1. - step_progress);
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

  void Interpolator::interpolateContacts(
    const double delay, const double timestep, const std::vector<std::vector<bool>> & cs, std::vector<bool> & c_interp)
  {
    // Compute the time knot corresponding to the current delay
    size_t step_nb = static_cast<size_t>(delay / timestep);
    step_nb = std::clamp(step_nb, 0UL, cs.size() - 1);

    // Set the output arg
    c_interp = cs[step_nb];
  }
} // namespace simple_mpc
