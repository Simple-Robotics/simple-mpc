
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

// Helper class to create the problem and run it
class TestKinoID
{
public:
  TestKinoID(RobotModelHandler model_handler_, KinodynamicsID::Settings settings_)
  : model_handler(model_handler_)
  , data_handler(model_handler)
  , settings(settings_)
  , solver(model_handler, dt, settings)
  , q(model_handler.getReferenceState().head(model_handler.getModel().nq))
  , dq(Eigen::VectorXd::Zero(model_handler.getModel().nv))
  , ddq(Eigen::VectorXd::Zero(model_handler.getModel().nv))
  , tau(Eigen::VectorXd::Zero(model_handler.getModel().nv - 6))
  {
  }

  void step()
  {
    // Solve
    solver.solve(t, q, dq, tau);
    solver.getAccelerations(ddq);

    // Integrate
    t += dt;
    q = pinocchio::integrate(model_handler.getModel(), q, (dq + ddq / 2. * dt) * dt);
    dq += ddq * dt;

    // Check common to all tests
    check_joint_limits();
  }

  void check_decreasing_error(std::string name, double error)
  {
    if (errors.count(name) == 0)
    {
      errors.insert({name, error});
      return; // no further check
    }
    BOOST_CHECK_LE(error, errors.at(name)); // Perform check
    errors.at(name) = error;                // Update value
  }

protected:
  void check_joint_limits()
  {
    const pinocchio::Model & model = model_handler.getModel();
    for (int i = 0; i < model.nv - 6; i++)
    {
      BOOST_CHECK_LE(q[7 + i], model.upperPositionLimit[7 + i]);
      BOOST_CHECK_GE(q[7 + i], model.lowerPositionLimit[7 + i]);
      BOOST_CHECK_LE(dq[6 + i], model.upperVelocityLimit[6 + i]);
      // Do not use lower velocity bound as TSID cannot handle it
      BOOST_CHECK_GE(dq[6 + i], -model.upperVelocityLimit[6 + i]);
      BOOST_CHECK_LE(tau[i], model.upperEffortLimit[6 + i]);
      BOOST_CHECK_GE(tau[i], model.lowerEffortLimit[6 + i]);
    }
  }

public:
  const RobotModelHandler model_handler;
  RobotDataHandler data_handler;
  KinodynamicsID::Settings settings;
  double dt = 1e-3;
  KinodynamicsID solver;

  double t = 0.;
  Eigen::VectorXd q;
  Eigen::VectorXd dq;
  Eigen::VectorXd ddq;
  Eigen::VectorXd tau;

  std::map<std::string, double> errors;
};

BOOST_AUTO_TEST_CASE(KinodynamicsID_postureTask)
{
  TestKinoID test(getSoloHandler(), KinodynamicsID::Settings().set_kp_posture(20.).set_w_posture(1.));

  // Easy access
  const RobotModelHandler & model_handler = test.model_handler;
  const size_t nq = model_handler.getModel().nq;
  const size_t nv = model_handler.getModel().nv;

  // Target state
  const Eigen::VectorXd q_target = model_handler.getReferenceState().head(nq);
  test.solver.setTarget(
    q_target, Eigen::VectorXd::Zero(nv), Eigen::VectorXd::Zero(nv), {false, false, false, false}, {});

  // Change initial state
  test.q = solo_q_start(model_handler);
  for (int i = 0; i < 1000; i++)
  {
    // Solve
    test.step();

    // compensate for free fall as we did not set any contact (we only care about joint posture)
    test.q.head(7) = q_target.head(7);
    test.dq.head(6).setZero();

    // Check error is decreasing
    Eigen::VectorXd delta_q = pinocchio::difference(model_handler.getModel(), test.q, q_target);
    const double error = delta_q.tail(nv - 6).norm(); // Consider only the posture not the free flyer
    test.check_decreasing_error("posture", error);
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
  std::vector<KinodynamicsID::TargetContactForce> f_target;
  for (int i = 0; i < 4; i++)
  {
    f_target.push_back(KinodynamicsID::TargetContactForce::Zero(3));
    f_target[i][2] = model_handler.getMass() * 9.81 / 4;
  }
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
  }
}

BOOST_AUTO_TEST_CASE(KinodynamicsID_allTasks)
{
  RobotModelHandler model_handler = getTalosModelHandler();
  RobotDataHandler data_handler(model_handler);
  const double dt = 1e-3;

  KinodynamicsID solver(
    model_handler, dt,
    KinodynamicsID::Settings()
      .set_kp_base(7.)
      .set_kp_posture(10.)
      .set_kp_contact(1.0)
      .set_w_base(10.0)
      .set_w_posture(10.0)
      .set_w_contact_force(.1)
      .set_w_contact_motion(1.0));

  const Eigen::VectorXd q_target = model_handler.getReferenceState().head(model_handler.getModel().nq);
  std::vector<KinodynamicsID::TargetContactForce> f_target;
  for (int i = 0; i < 2; i++)
  {
    f_target.push_back(KinodynamicsID::TargetContactForce::Zero(6));
    f_target[i][2] = model_handler.getMass() * 9.81 / 4;
  }
  solver.setTarget(
    q_target, Eigen::VectorXd::Zero(model_handler.getModel().nv), Eigen::VectorXd::Zero(model_handler.getModel().nv),
    {true, true, true, true}, f_target);

  double t = 0;
  Eigen::VectorXd q = model_handler.getReferenceState().head(model_handler.getModel().nq);
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
  }
}

BOOST_AUTO_TEST_SUITE_END()
