
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>

#include "simple-mpc/inverse-dynamics.hpp"
#include "simple-mpc/robot-handler.hpp"
#include "test_utils.cpp"

BOOST_AUTO_TEST_SUITE(inverse_dynamics)

using namespace simple_mpc;

Eigen::VectorXd solo_q_start(const RobotModelHandler & model_handler)
{
  Eigen::VectorXd q_start = model_handler.getReferenceState().head(model_handler.getModel().nq);
  for (int l = 0; l < 4; l++)
  {
    q_start[7 + 3 * l + 1] = 0.9;
    q_start[7 + 3 * l + 2] = -1.8;
  }
  q_start[0] = 0.01;
  q_start[1] = 0.01;
  q_start[2] = 0.21;

  return q_start;
}

void check_joint_limits(
  const RobotModelHandler & model_handler,
  const Eigen::VectorXd & q,
  const Eigen::VectorXd & v,
  const Eigen::VectorXd & tau)
{
  const pinocchio::Model & model = model_handler.getModel();
  for (int i = 0; i < model.nv - 6; i++)
  {
    BOOST_CHECK_LE(q[7 + i], model.upperPositionLimit[7 + i]);
    BOOST_CHECK_GE(q[7 + i], model.lowerPositionLimit[7 + i]);
    BOOST_CHECK_LE(v[6 + i], model.upperVelocityLimit[6 + i]);
    // Do not use lower velocity bound as TSID cannot handle it
    BOOST_CHECK_GE(v[6 + i], -model.upperVelocityLimit[6 + i]);
    BOOST_CHECK_LE(tau[i], model.upperEffortLimit[6 + i]);
    BOOST_CHECK_GE(tau[i], model.lowerEffortLimit[6 + i]);
  }
}

BOOST_AUTO_TEST_CASE(KinodynamicsID_postureTask)
{
  RobotModelHandler model_handler = getSoloHandler();
  RobotDataHandler data_handler(model_handler);
  const double dt = 1e-3;

  KinodynamicsID solver(
    model_handler, dt, KinodynamicsID::Settings().set_kp_posture(20.).set_w_posture(1.)); // only a posture task

  const Eigen::VectorXd q_target = model_handler.getReferenceState().head(model_handler.getModel().nq);

  solver.setTarget(
    q_target, Eigen::VectorXd::Zero(model_handler.getModel().nv), Eigen::VectorXd::Zero(model_handler.getModel().nv),
    {false, false, false, false}, Eigen::MatrixXd::Zero(4, 3));

  double t = 0;
  Eigen::VectorXd q = solo_q_start(model_handler);
  Eigen::VectorXd v = Eigen::VectorXd::Random(model_handler.getModel().nv);
  Eigen::VectorXd a = Eigen::VectorXd::Random(model_handler.getModel().nv);
  Eigen::VectorXd tau = Eigen::VectorXd::Zero(model_handler.getModel().nv - 6);

  Eigen::VectorXd error = 1e12 * Eigen::VectorXd::Ones(model_handler.getModel().nv);

  for (int i = 0; i < 10000; i++)
  {
    // Solve and get solution
    solver.solve(t, q, v, tau);
    solver.getAccelerations(a);

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

    // Check joints limits
    check_joint_limits(model_handler, q, v, tau);
  }
}

BOOST_AUTO_TEST_CASE(KinodynamicsID_contact)
{
  RobotModelHandler model_handler = getSoloHandler();
  RobotDataHandler data_handler(model_handler);
  const double dt = 1e-3;

  KinodynamicsID solver(
    model_handler, dt,
    KinodynamicsID::Settings()
      .set_kp_base(10.)
      .set_kp_posture(10.0)
      .set_kp_contact(10.0)
      .set_w_base(10.)
      .set_w_posture(1.0)
      .set_w_contact_motion(1.0)
      .set_w_contact_force(1.0));

  const Eigen::VectorXd q_target = model_handler.getReferenceState().head(model_handler.getModel().nq);
  Eigen::MatrixXd f_target = Eigen::MatrixXd::Zero(4, 3);
  f_target(0, 2) = model_handler.getMass() * 9.81 / 4;
  f_target(1, 2) = model_handler.getMass() * 9.81 / 4;
  f_target(2, 2) = model_handler.getMass() * 9.81 / 4;
  f_target(3, 2) = model_handler.getMass() * 9.81 / 4;

  solver.setTarget(
    q_target, Eigen::VectorXd::Zero(model_handler.getModel().nv), Eigen::VectorXd::Zero(model_handler.getModel().nv),
    {true, true, true, true}, f_target);

  double t = 0;
  Eigen::VectorXd q = solo_q_start(model_handler);
  Eigen::VectorXd v = Eigen::VectorXd::Random(model_handler.getModel().nv);
  Eigen::VectorXd a = Eigen::VectorXd::Random(model_handler.getModel().nv);
  Eigen::VectorXd tau = Eigen::VectorXd::Zero(model_handler.getModel().nv - 6);

  // Let the robot stabilize
  const int N_STEP_ON_GROUND = 6000;
  const int N_STEP_FREE_FALL = 2000;
  for (int i = 0; i < N_STEP_ON_GROUND + N_STEP_FREE_FALL; i++)
  {
    // Solve and get solution
    solver.solve(t, q, v, tau);
    solver.getAccelerations(a);

    // Integrate
    t += dt;
    q = pinocchio::integrate(model_handler.getModel(), q, (v + a / 2. * dt) * dt);
    v += a * dt;
    if (i == N_STEP_ON_GROUND)
    {
      // Robot had time to reach permanent regime, is it stable on ground ?
      BOOST_CHECK_SMALL(a.head(3).norm(), 1e-4);
      BOOST_CHECK_SMALL(v.head(3).norm(), 1e-4);

      // Remove contacts
      solver.setTarget(
        q_target, Eigen::VectorXd::Zero(model_handler.getModel().nv),
        Eigen::VectorXd::Zero(model_handler.getModel().nv), {false, false, false, false}, f_target);
    }
    if (i == N_STEP_ON_GROUND + N_STEP_FREE_FALL - 1)
    {
      // Robot had time to reach permanent regime, is it robot free falling ?
      BOOST_CHECK_SMALL(a.head(3).norm() - model_handler.getModel().gravity.linear().norm(), 0.01);
    }

    // Check joints limits
    check_joint_limits(model_handler, q, v, tau);
  }
}

BOOST_AUTO_TEST_CASE(KinodynamicsID_baseTask)
{
  RobotModelHandler model_handler = getSoloHandler();
  RobotDataHandler data_handler(model_handler);
  const double dt = 1e-3;

  KinodynamicsID solver(
    model_handler, dt,
    KinodynamicsID::Settings()
      .set_kp_base(7.)
      .set_kp_contact(10.0)
      .set_w_base(100.0)
      .set_w_contact_force(1.0)
      .set_w_contact_motion(1.0));

  // No need to set target as KinodynamicsID sets it by default to reference state
  const Eigen::VectorXd q_target = model_handler.getReferenceState().head(model_handler.getModel().nq);

  double t = 0;
  Eigen::VectorXd q = solo_q_start(model_handler);
  Eigen::VectorXd v = Eigen::VectorXd::Random(model_handler.getModel().nv);
  Eigen::VectorXd a = Eigen::VectorXd::Random(model_handler.getModel().nv);
  Eigen::VectorXd tau = Eigen::VectorXd::Zero(model_handler.getModel().nv - 6);

  Eigen::VectorXd error = 1e12 * Eigen::VectorXd::Ones(6);
  const int N_STEP = 10000;
  for (int i = 0; i < N_STEP; i++)
  {
    // Solve and get solution
    solver.solve(t, q, v, tau);
    solver.getAccelerations(a);

    // Integrate
    t += dt;
    q = pinocchio::integrate(model_handler.getModel(), q, (v + a / 2. * dt) * dt);
    v += a * dt;

    // Check error is decreasing
    Eigen::VectorXd new_error = pinocchio::difference(model_handler.getModel(), q, q_target).head(6);
    if (i > N_STEP / 10) // Weird transitional phenomenon at first ...
      BOOST_CHECK(
        new_error.norm() < error.norm() || new_error.norm() < 2e-2); // Either strictly decreasing or close to target
    if (i > 9 * N_STEP / 10)
      BOOST_CHECK(new_error.norm() < 2e-2); // Should have converged by now

    error = new_error;
    // Check joints limits
    check_joint_limits(model_handler, q, v, tau);
  }
}

BOOST_AUTO_TEST_CASE(KinodynamicsID_allTasks)
{
  RobotModelHandler model_handler = getSoloHandler();
  RobotDataHandler data_handler(model_handler);
  const double dt = 1e-3;

  KinodynamicsID solver(
    model_handler, dt,
    KinodynamicsID::Settings()
      .set_kp_base(7.)
      .set_kp_posture(10.)
      .set_kp_contact(10.0)
      .set_w_base(100.0)
      .set_w_posture(1.0)
      .set_w_contact_force(1.0)
      .set_w_contact_motion(1.0));

  const Eigen::VectorXd q_target = model_handler.getReferenceState().head(model_handler.getModel().nq);
  Eigen::MatrixXd f_target = Eigen::MatrixXd::Zero(4, 3);
  f_target(0, 2) = model_handler.getMass() * 9.81 / 4;
  f_target(1, 2) = model_handler.getMass() * 9.81 / 4;
  f_target(2, 2) = model_handler.getMass() * 9.81 / 4;
  f_target(3, 2) = model_handler.getMass() * 9.81 / 4;

  solver.setTarget(
    q_target, Eigen::VectorXd::Zero(model_handler.getModel().nv), Eigen::VectorXd::Zero(model_handler.getModel().nv),
    {true, true, true, true}, f_target);

  double t = 0;
  Eigen::VectorXd q = solo_q_start(model_handler);
  Eigen::VectorXd v = Eigen::VectorXd::Random(model_handler.getModel().nv);
  Eigen::VectorXd a = Eigen::VectorXd::Random(model_handler.getModel().nv);
  Eigen::VectorXd tau = Eigen::VectorXd::Zero(model_handler.getModel().nv - 6);

  Eigen::VectorXd error = 1e12 * Eigen::VectorXd::Ones(model_handler.getModel().nv);

  const int N_STEP = 10000;
  for (int i = 0; i < N_STEP; i++)
  {
    // Solve and get solution
    solver.solve(t, q, v, tau);
    solver.getAccelerations(a);

    // Integrate
    t += dt;
    q = pinocchio::integrate(model_handler.getModel(), q, (v + a / 2. * dt) * dt);
    v += a * dt;

    // Check error is decreasing
    Eigen::VectorXd new_error = pinocchio::difference(model_handler.getModel(), q, q_target);
    BOOST_CHECK_LE(new_error.norm(), error.norm());
    error = new_error;

    // Check joints limits
    check_joint_limits(model_handler, q, v, tau);
  }
}

BOOST_AUTO_TEST_SUITE_END()
