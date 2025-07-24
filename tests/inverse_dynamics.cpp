
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>

#include "simple-mpc/inverse-dynamics.hpp"
#include "simple-mpc/robot-handler.hpp"
#include "test_utils.cpp"

BOOST_AUTO_TEST_SUITE(inverse_dynamics)

using namespace simple_mpc;

BOOST_AUTO_TEST_CASE(KinodynamicsID_postureTask)
{
  RobotModelHandler model_handler = getSoloHandler();
  RobotDataHandler data_handler(model_handler);

  KinodynamicsID solver(
    model_handler, KinodynamicsID::Settings::Default().set_w_base(0.).set_w_contact_force(0.).set_w_contact_motion(0.));

  const Eigen::VectorXd q_target = model_handler.getReferenceState().head(model_handler.getModel().nq);

  solver.setTarget(
    q_target, Eigen::VectorXd::Zero(model_handler.getModel().nv), Eigen::VectorXd::Zero(model_handler.getModel().nv),
    {false, false, false, false}, Eigen::VectorXd::Zero(4 * 3));

  double t = 0;
  double dt = 1e-3;
  Eigen::VectorXd q = pinocchio::randomConfiguration(model_handler.getModel());
  q.head(7) = q_target.head(7);
  Eigen::VectorXd v = Eigen::VectorXd::Random(model_handler.getModel().nv);
  Eigen::VectorXd a = Eigen::VectorXd::Random(model_handler.getModel().nv);
  Eigen::VectorXd tau = Eigen::VectorXd::Zero(model_handler.getModel().nv - 6);

  Eigen::VectorXd error = 1e12 * Eigen::VectorXd::Ones(model_handler.getModel().nv);

  for (int i = 0; i < 10000; i++)
  {
    // Solve and get solution
    solver.solve(t, q, v, tau);
    a = solver.getAccelerations();

    // Integrate
    t += dt;
    q = pinocchio::integrate(model_handler.getModel(), q, (v + a / 2. * dt) * dt);
    v += a * dt;

    // compensate for free fall as we only care about joint posture
    q.head(7) = q_target.head(7);
    v.head(6).setZero();

    // Check error is decreasing
    Eigen::VectorXd new_error = pinocchio::difference(model_handler.getModel(), q, q_target);
    BOOST_CHECK_LE(new_error.norm(), error.norm());
    error = new_error;
  }
}

BOOST_AUTO_TEST_CASE(KinodynamicsID_allTasks)
{
  RobotModelHandler model_handler = getSoloHandler();
  RobotDataHandler data_handler(model_handler);

  KinodynamicsID solver(model_handler);

  const Eigen::VectorXd q_target = model_handler.getReferenceState().head(model_handler.getModel().nq);
  Eigen::VectorXd f_target = Eigen::VectorXd::Zero(4 * 3);
  f_target[2] = model_handler.getMass() * 9.81 / 4;
  f_target[5] = model_handler.getMass() * 9.81 / 4;
  f_target[8] = model_handler.getMass() * 9.81 / 4;
  f_target[11] = model_handler.getMass() * 9.81 / 4;

  solver.setTarget(
    q_target, Eigen::VectorXd::Zero(model_handler.getModel().nv), Eigen::VectorXd::Zero(model_handler.getModel().nv),
    {true, true, true, true}, f_target);

  double t = 0;
  double dt = 1e-3;
  Eigen::VectorXd q = pinocchio::randomConfiguration(model_handler.getModel());
  q.head(7) = q_target.head(7);
  Eigen::VectorXd v = Eigen::VectorXd::Random(model_handler.getModel().nv);
  Eigen::VectorXd a = Eigen::VectorXd::Random(model_handler.getModel().nv);
  Eigen::VectorXd tau = Eigen::VectorXd::Zero(model_handler.getModel().nv - 6);

  Eigen::VectorXd error = 1e12 * Eigen::VectorXd::Ones(model_handler.getModel().nv);

  for (int i = 0; i < 1000; i++)
  {
    // Solve and get solution
    solver.solve(t, q, v, tau);
    a = solver.getAccelerations();

    // Integrate
    t += dt;
    q = pinocchio::integrate(model_handler.getModel(), q, (v + a / 2. * dt) * dt);
    v += a * dt;

    // Check error is decreasing
    Eigen::VectorXd new_error = pinocchio::difference(model_handler.getModel(), q, q_target);
    BOOST_CHECK_LE(new_error.norm(), error.norm());
    error = new_error;
  }
}

BOOST_AUTO_TEST_SUITE_END()
