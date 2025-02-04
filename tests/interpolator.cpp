
#include <boost/test/unit_test.hpp>

#include "simple-mpc/interpolator.hpp"
#include "test_utils.cpp"

BOOST_AUTO_TEST_SUITE(interpolator)

using namespace simple_mpc;

BOOST_AUTO_TEST_CASE(interpolate)
{
  long nx = 27;
  long nu = 12;
  long nv = 18;
  long nf = 12;
  double MPC_timestep = 0.01;
  Interpolator interpolator = Interpolator(nx, nv, nu, nf, MPC_timestep);

  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  std::vector<Eigen::VectorXd> ddqs;
  std::vector<Eigen::VectorXd> forces;
  for (std::size_t i = 0; i < 4; i++)
  {
    xs.push_back(Eigen::Vector<double, 27>::Random());
    us.push_back(Eigen::Vector<double, 12>::Random());
    ddqs.push_back(Eigen::Vector<double, 18>::Random());
    forces.push_back(Eigen::Vector<double, 12>::Random());
  }
  double delay = 0.02;

  interpolator.interpolate(delay, xs, us, ddqs, forces);

  BOOST_CHECK(xs[2].isApprox(interpolator.x_interpolated_));
  BOOST_CHECK(us[2].isApprox(interpolator.u_interpolated_));
  BOOST_CHECK(ddqs[2].isApprox(interpolator.a_interpolated_));
  BOOST_CHECK(forces[2].isApprox(interpolator.forces_interpolated_));
}

BOOST_AUTO_TEST_SUITE_END()
