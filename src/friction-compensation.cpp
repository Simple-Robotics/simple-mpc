#include "simple-mpc/friction-compensation.hpp"

namespace simple_mpc
{

  FrictionCompensation::FrictionCompensation(const Model & model, const bool with_free_flyer)
  {
    if (with_free_flyer)
    {
      // Ignore universe and root joints
      nu_ = model.njoints - 2;
    }
    else
    {
      // Ignore universe joint
      nu_ = model.njoints - 1;
    }
    dry_friction_ = model.friction.tail(nu_);
    viscuous_friction_ = model.damping.tail(nu_);
  }

  void FrictionCompensation::computeFriction(Eigen::Ref<const VectorXd> velocity, Eigen::Ref<VectorXd> torque)
  {
    assert(("Velocity size must be equal to actuation size", velocity.size() == nu_));
    assert(("Torque size must be equal to actuation size", torque.size() == nu_));
    torque += viscuous_friction_.cwiseProduct(velocity)
              + dry_friction_.cwiseProduct(velocity.unaryExpr(std::function(signFunction)));
  }

  double FrictionCompensation::signFunction(double x)
  {
    return (x > 0) - (x < 0);
  }

} // namespace simple_mpc
