#include <boost/test/unit_test.hpp>

#include "simple-mpc/centroidal-dynamics.hpp"
#include "simple-mpc/fulldynamics.hpp"
#include "simple-mpc/fwd.hpp"
#include "simple-mpc/kinodynamics.hpp"
#include "simple-mpc/mpc.hpp"
#include "simple-mpc/robot-handler.hpp"
#include "test_utils.cpp"

BOOST_AUTO_TEST_SUITE(mpc)

using namespace simple_mpc;

BOOST_AUTO_TEST_CASE(mpc_fulldynamics)
{
  RobotModelHandler model_handler = getTalosModelHandler();
  RobotDataHandler data_handler(model_handler);

  FullDynamicsSettings settings = getFullDynamicsSettings(model_handler);
  auto problem = std::make_shared<FullDynamicsOCP>(settings, model_handler);
  FullDynamicsOCP & fdproblem = *problem;

  const size_t T = 100;
  fdproblem.createProblem(model_handler.getReferenceState(), T, 6, -settings.gravity[2], true);

  MPCSettings mpc_settings;
  mpc_settings.max_iters = 1;

  mpc_settings.support_force = -model_handler.getMass() * settings.gravity[2];

  mpc_settings.TOL = 1e-6;
  mpc_settings.mu_init = 1e-8;
  mpc_settings.num_threads = 1;

  mpc_settings.swing_apex = 0.1;
  mpc_settings.T_fly = 80;
  mpc_settings.T_contact = 20;
  mpc_settings.timestep = 0.01;

  MPC mpc = MPC(mpc_settings, problem);

  BOOST_CHECK_EQUAL(mpc.xs_.size(), T + 1);
  BOOST_CHECK_EQUAL(mpc.us_.size(), T);

  std::vector<std::map<std::string, bool>> contact_states;
  for (std::size_t i = 0; i < 10; i++)
  {
    std::map<std::string, bool> contact_state;
    contact_state.insert({model_handler.getFootName(0), true});
    contact_state.insert({model_handler.getFootName(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 50; i++)
  {
    std::map<std::string, bool> contact_state;
    contact_state.insert({model_handler.getFootName(0), true});
    contact_state.insert({model_handler.getFootName(1), false});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 10; i++)
  {
    std::map<std::string, bool> contact_state;
    contact_state.insert({model_handler.getFootName(0), true});
    contact_state.insert({model_handler.getFootName(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 50; i++)
  {
    std::map<std::string, bool> contact_state;
    contact_state.insert({model_handler.getFootName(0), false});
    contact_state.insert({model_handler.getFootName(1), true});
    contact_states.push_back(contact_state);
  }

  mpc.generateCycleHorizon(contact_states);

  BOOST_CHECK_EQUAL(mpc.foot_takeoff_times_.at("left_sole_link")[0], 170);
  BOOST_CHECK_EQUAL(mpc.foot_takeoff_times_.at("right_sole_link")[0], 110);
  BOOST_CHECK_EQUAL(mpc.foot_land_times_.at("left_sole_link")[0], 219);
  BOOST_CHECK_EQUAL(mpc.foot_land_times_.at("right_sole_link")[0], 160);
  for (std::size_t i = 0; i < 10; i++)
  {
    mpc.iterate(model_handler.getReferenceState());
  }

  BOOST_CHECK_EQUAL(mpc.foot_takeoff_times_.at("left_sole_link")[0], 160);
  BOOST_CHECK_EQUAL(mpc.foot_takeoff_times_.at("right_sole_link")[0], 100);
  BOOST_CHECK_EQUAL(mpc.foot_land_times_.at("left_sole_link")[0], 209);
  BOOST_CHECK_EQUAL(mpc.foot_land_times_.at("right_sole_link")[0], 150);

  Eigen::VectorXd xdot = mpc.getStateDerivative(0);

  std::vector<Eigen::Vector3d> forces = mpc.getContactForces(0);
}

BOOST_AUTO_TEST_CASE(mpc_kinodynamics)
{
  RobotModelHandler model_handler = getTalosModelHandler();
  RobotDataHandler data_handler(model_handler);

  KinodynamicsSettings settings = getKinodynamicsSettings(model_handler);
  auto problem = std::make_shared<KinodynamicsOCP>(settings, model_handler);
  KinodynamicsOCP & kinoproblem = *problem;
  const std::size_t T = 100;
  const double support_force = -model_handler.getMass() * settings.gravity[2];
  Eigen::VectorXd f1(6);
  f1 << 0, 0, support_force, 0, 0, 0;

  kinoproblem.createProblem(model_handler.getReferenceState(), T, 6, -settings.gravity[2], true);

  MPCSettings mpc_settings;
  mpc_settings.max_iters = 1;

  mpc_settings.support_force = support_force;

  mpc_settings.TOL = 1e-6;
  mpc_settings.mu_init = 1e-8;
  mpc_settings.num_threads = 8;

  mpc_settings.swing_apex = 0.1;
  mpc_settings.T_fly = 80;
  mpc_settings.T_contact = 20;
  mpc_settings.timestep = 0.01;

  MPC mpc = MPC(mpc_settings, problem);

  BOOST_CHECK_EQUAL(mpc.xs_.size(), T + 1);
  BOOST_CHECK_EQUAL(mpc.us_.size(), T);

  std::vector<std::map<std::string, bool>> contact_states;
  // std::vector<std::vector<bool>> contact_states;
  for (std::size_t i = 0; i < 10; i++)
  {
    std::map<std::string, bool> contact_state;
    contact_state.insert({model_handler.getFootName(0), true});
    contact_state.insert({model_handler.getFootName(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 50; i++)
  {
    std::map<std::string, bool> contact_state;
    contact_state.insert({model_handler.getFootName(0), true});
    contact_state.insert({model_handler.getFootName(1), false});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 10; i++)
  {
    std::map<std::string, bool> contact_state;
    contact_state.insert({model_handler.getFootName(0), true});
    contact_state.insert({model_handler.getFootName(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 50; i++)
  {
    std::map<std::string, bool> contact_state;
    contact_state.insert({model_handler.getFootName(0), false});
    contact_state.insert({model_handler.getFootName(1), true});
    contact_states.push_back(contact_state);
  }

  mpc.generateCycleHorizon(contact_states);

  for (std::size_t i = 0; i < 10; i++)
  {
    mpc.iterate(model_handler.getReferenceState());
  }

  Eigen::VectorXd xdot = mpc.getStateDerivative(0);
}

BOOST_AUTO_TEST_CASE(mpc_centroidal)
{
  RobotModelHandler model_handler = getTalosModelHandler();
  RobotDataHandler data_handler(model_handler);

  CentroidalSettings settings = getCentroidalSettings();
  auto problem = std::make_shared<CentroidalOCP>(settings, model_handler);
  CentroidalOCP & centproblem = *problem;

  std::vector<std::string> contact_names = {"left_sole_link", "right_sole_link"};
  const double support_force = -model_handler.getMass() * settings.gravity[2];
  const std::size_t T = 100;
  Eigen::VectorXd f1(6);
  f1 << 0, 0, support_force / 2., 0, 0, 0;
  Eigen::VectorXd x_multibody = model_handler.getReferenceState();

  centproblem.createProblem(data_handler.getCentroidalState(), T, 6, -settings.gravity[2], false);

  MPCSettings mpc_settings;
  mpc_settings.max_iters = 1;

  mpc_settings.support_force = support_force;

  mpc_settings.TOL = 1e-6;
  mpc_settings.mu_init = 1e-8;
  mpc_settings.num_threads = 8;

  mpc_settings.swing_apex = 0.1;
  mpc_settings.T_fly = 80;
  mpc_settings.T_contact = 20;
  mpc_settings.timestep = 0.01;

  MPC mpc = MPC(mpc_settings, problem);

  BOOST_CHECK_EQUAL(mpc.xs_.size(), T + 1);
  BOOST_CHECK_EQUAL(mpc.us_.size(), T);

  std::vector<std::map<std::string, bool>> contact_states;
  for (std::size_t i = 0; i < 10; i++)
  {
    std::map<std::string, bool> contact_state;
    contact_state.insert({model_handler.getFootName(0), true});
    contact_state.insert({model_handler.getFootName(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 50; i++)
  {
    std::map<std::string, bool> contact_state;
    contact_state.insert({model_handler.getFootName(0), true});
    contact_state.insert({model_handler.getFootName(1), false});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 10; i++)
  {
    std::map<std::string, bool> contact_state;
    contact_state.insert({model_handler.getFootName(0), true});
    contact_state.insert({model_handler.getFootName(1), true});
    contact_states.push_back(contact_state);
  }
  for (std::size_t i = 0; i < 50; i++)
  {
    std::map<std::string, bool> contact_state;
    contact_state.insert({model_handler.getFootName(0), false});
    contact_state.insert({model_handler.getFootName(1), true});
    contact_states.push_back(contact_state);
  }

  mpc.generateCycleHorizon(contact_states);

  for (std::size_t i = 0; i < 10; i++)
  {
    mpc.iterate(x_multibody);
  }

  Eigen::VectorXd xdot = mpc.getStateDerivative(0);
}

BOOST_AUTO_TEST_SUITE_END()
