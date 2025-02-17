
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
  Interpolator interpolator = Interpolator(model);

  // Test configuration interpolation method
  std::vector<Eigen::VectorXd> qs;
  for (std::size_t i = 0; i < 4; i++)
  {
    Eigen::VectorXd q0(model.nq);
    Eigen::VectorXd q1(model.nq);
    q0 = pinocchio::neutral(model);
    Eigen::VectorXd dq(model.nv);
    dq.setRandom();
    pinocchio::integrate(model, q0, dq, q1);

    qs.push_back(q1);
  }
  double delay = 0.02;

  Eigen::VectorXd q_interp(model.nq);
  interpolator.interpolateConfiguration(delay, timestep, qs, q_interp);

  BOOST_CHECK(qs[2].isApprox(q_interp));

  delay = 0.5;
  interpolator.interpolateConfiguration(delay, timestep, qs, q_interp);
  BOOST_CHECK(qs.back().isApprox(q_interp));

  delay = 0.005;
  interpolator.interpolateConfiguration(delay, timestep, qs, q_interp);
  Eigen::VectorXd q_interp2(model.nq);
  Eigen::VectorXd dq(model.nv);
  pinocchio::difference(model, qs[0], qs[1], dq);
  pinocchio::integrate(model, qs[0], dq * 0.5, q_interp2);

  BOOST_CHECK(q_interp2.isApprox(q_interp));

  std::vector<Eigen::VectorXd> qs2;
  for (std::size_t i = 0; i < 2; i++)
  {
    qs2.push_back(qs[0]);
  }
  interpolator.interpolateConfiguration(delay, timestep, qs2, q_interp);
  BOOST_CHECK(qs2[0].isApprox(q_interp));

  // Test linear interpolation method
  std::vector<Eigen::VectorXd> vs;
  for (std::size_t i = 0; i < 4; i++)
  {
    Eigen::VectorXd v0(model.nv);
    v0.setRandom();

    vs.push_back(v0);
  }
  delay = 0.02;
  Eigen::VectorXd v_interp(model.nv);
  interpolator.interpolateLinear(delay, timestep, vs, v_interp);
  BOOST_CHECK(vs[2].isApprox(v_interp));

  delay = 0.5;
  interpolator.interpolateLinear(delay, timestep, vs, v_interp);
  BOOST_CHECK(vs.back().isApprox(v_interp));

  delay = 0.005;
  interpolator.interpolateLinear(delay, timestep, vs, v_interp);
  Eigen::VectorXd v_interp2(model.nv);
  v_interp2 = (vs[0] + vs[1]) * 0.5;

  BOOST_CHECK(v_interp2.isApprox(v_interp));

  std::vector<Eigen::VectorXd> vs2;
  for (std::size_t i = 0; i < 2; i++)
  {
    vs2.push_back(vs[0]);
  }
  interpolator.interpolateLinear(delay, timestep, vs2, v_interp);
  BOOST_CHECK(vs2[0].isApprox(v_interp));
}

BOOST_AUTO_TEST_SUITE_END()
