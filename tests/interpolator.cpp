
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

  delay = 0.5;
  interpolator.interpolate(delay, timestep, xs, x_interp);
  BOOST_CHECK(xs.back().isApprox(x_interp));

  delay = 0.005;
  interpolator.interpolate(delay, timestep, xs, x_interp);
  Eigen::VectorXd x_interp2(model.nq + model.nv);
  Eigen::VectorXd dq(model.nv);
  pinocchio::difference(model, xs[0].head(model.nq), xs[1].head(model.nq), dq);
  pinocchio::integrate(model, xs[0].head(model.nq), dq * 0.5, x_interp2.head(model.nq));
  x_interp2.tail(model.nv) = (xs[0].tail(model.nv) + xs[1].tail(model.nv)) * 0.5;

  BOOST_CHECK(x_interp2.isApprox(x_interp));
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

  delay = 0.5;
  interpolator.interpolate(delay, timestep, vs, v_interp);
  BOOST_CHECK(vs.back().isApprox(v_interp));

  delay = 0.005;
  interpolator.interpolate(delay, timestep, vs, v_interp);
  Eigen::VectorXd v_interp2(model.nv);
  v_interp2 = (vs[0] + vs[1]) * 0.5;

  BOOST_CHECK(v_interp2.isApprox(v_interp));
}

BOOST_AUTO_TEST_SUITE_END()
