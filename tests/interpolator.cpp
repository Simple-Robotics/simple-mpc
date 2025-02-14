
#include <boost/test/unit_test.hpp>

#include "simple-mpc/interpolator.hpp"
#include "test_utils.cpp"

BOOST_AUTO_TEST_SUITE(interpolator)

using namespace simple_mpc;

BOOST_AUTO_TEST_CASE(interpolate)
{
  Model model;
  // Load pinocchio model from example robot data
  const std::string urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/robots/solo12.urdf";
  const std::string srdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/srdf/solo.srdf";

  pinocchio::urdf::buildModel(urdf_path, JointModelFreeFlyer(), model);
  double timestep = 0.01;
  StateInterpolator interpolator = StateInterpolator(model);

  std::vector<Eigen::VectorXd> xs;
  for (std::size_t i = 0; i < 4; i++)
  {
    Eigen::VectorXd x0(model.nq + model.nv);
    x0.tail(model.nv).setRandom();
    x0.head(model.nq) = pinocchio::neutral(model);
    Eigen::VectorXd dq(model.nv);
    dq.setRandom();
    pinocchio::integrate(model, x0.head(model.nq), dq, x0.head(model.nq));

    xs.push_back(x0);
  }
  double delay = 0.02;

  Eigen::VectorXd x_interp(model.nq + model.nv);
  interpolator.interpolate(delay, timestep, xs, x_interp);

  BOOST_CHECK(xs[2].isApprox(x_interp));
}

BOOST_AUTO_TEST_CASE(linear_interpolate)
{
  Model model;
  // Load pinocchio model from example robot data
  const std::string urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/robots/solo12.urdf";
  const std::string srdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/srdf/solo.srdf";

  pinocchio::urdf::buildModel(urdf_path, JointModelFreeFlyer(), model);
  double timestep = 0.01;
  LinearInterpolator interpolator = LinearInterpolator((size_t)model.nv);

  std::vector<Eigen::VectorXd> vs;
  for (std::size_t i = 0; i < 4; i++)
  {
    Eigen::VectorXd v0(model.nv);
    v0.setRandom();

    vs.push_back(v0);
  }
  double delay = 0.02;

  Eigen::VectorXd v_interp(model.nv);
  interpolator.interpolate(delay, timestep, vs, v_interp);

  BOOST_CHECK(vs[2].isApprox(v_interp));
}

BOOST_AUTO_TEST_SUITE_END()
