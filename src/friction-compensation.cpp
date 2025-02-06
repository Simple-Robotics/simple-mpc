#include "simple-mpc/friction-compensation.hpp"

namespace simple_mpc
{

  FrictionCompensation::FrictionCompensation(const Model & model, const long actuation_size)
  {
    corrected_torque_.resize(actuation_size);
    dry_friction_ = model.lowerDryFrictionLimit;
    viscuous_friction_ = model.damping;
  }

  void FrictionCompensation::computeFriction(const Eigen::VectorXd & velocity, const Eigen::VectorXd & torque)
  {
    if (velocity.size() != corrected_torque_.size())
    {
      throw std::runtime_error("Velocity has wrong size");
    }
    if (torque.size() != corrected_torque_.size())
    {
      throw std::runtime_error("Torque has wrong size");
    }
    corrected_torque_ = torque + viscuous_friction_.cwiseProduct(velocity)
                        + dry_friction_.cwiseProduct(velocity.unaryExpr(std::function(signFunction)));
  }

  double FrictionCompensation::signFunction(double x)
  {
    if (x > 0)
      return 1.0;
    else if (x == 0)
      return 0.0;
    else
      return -1.0;
  }

} // namespace simple_mpc
