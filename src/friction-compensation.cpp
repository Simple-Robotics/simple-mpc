#include "simple-mpc/friction-compensation.hpp"
#include <iostream>
namespace simple_mpc
{

  FrictionCompensation::FrictionCompensation(const Model & model, const bool with_free_flyer)
  {
    if (with_free_flyer)
    {
      // Ignore free flyer joint
      nu_ = model.nv - 6;
    }
    else
    {
      // If no free flyer
      nu_ = model.nv;
    }
    dry_friction_ = model.friction.tail(nu_);
    viscuous_friction_ = model.damping.tail(nu_);
  }

  void FrictionCompensation::computeFriction(Eigen::Ref<const VectorXd> velocity, Eigen::Ref<VectorXd> torque)
  {
    if (velocity.size() != nu_)
      throw std::runtime_error("Velocity has wrong size");
    if (torque.size() != nu_)
      throw std::runtime_error("Torque has wrong size");

    torque += viscuous_friction_.cwiseProduct(velocity)
              + dry_friction_.cwiseProduct(velocity.unaryExpr(std::function(signFunction)));
  }

  double FrictionCompensation::signFunction(double x)
  {
    return (x > 0) - (x < 0);
  }

} // namespace simple_mpc
