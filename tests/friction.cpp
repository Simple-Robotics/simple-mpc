
#include <boost/test/unit_test.hpp>

#include "simple-mpc/friction-compensation.hpp"
#include "test_utils.cpp"

BOOST_AUTO_TEST_SUITE(friction)

using namespace simple_mpc;

BOOST_AUTO_TEST_CASE(dry_viscuous_friction)
{
  Model model;

  const std::string urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/robots/solo12.urdf";

  pinocchio::urdf::buildModel(urdf_path, JointModelFreeFlyer(), model);
  long nu = 12;
  model.friction = Eigen::VectorXd::Constant(nu, .5);
  model.damping = Eigen::VectorXd::Constant(nu, .05);

  FrictionCompensation friction = FrictionCompensation(model, nu);

  BOOST_CHECK_EQUAL(model.friction, friction.dry_friction_);
  BOOST_CHECK_EQUAL(model.damping, friction.viscuous_friction_);

  Eigen::VectorXd velocity = Eigen::VectorXd::Random(nu);
  Eigen::VectorXd torque = Eigen::VectorXd::Random(nu);

  friction.computeFriction(velocity, torque);

  Eigen::VectorXd ctorque(nu);
  for (long i = 0; i < nu; i++)
  {
    double sgn = (velocity[i] > 0) - (velocity[i] < 0);
    ctorque[i] = torque[i] + model.friction[i] * sgn + model.damping[i] * velocity[i];
  }

  BOOST_CHECK(friction.corrected_torque_.isApprox(ctorque));
}

BOOST_AUTO_TEST_SUITE_END()
