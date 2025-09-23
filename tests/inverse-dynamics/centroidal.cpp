
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include "simple-mpc/inverse-dynamics/centroidal.hpp"
#include "simple-mpc/robot-handler.hpp"
#include "test_utils.cpp"

BOOST_AUTO_TEST_SUITE(inverse_dynamics_centroidal)

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
class TestCentroidalID
{
public:
  TestCentroidalID(RobotModelHandler model_handler_, CentroidalID::Settings settings_)
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
    step_i += 1;
    t += dt;
    q = pinocchio::integrate(model_handler.getModel(), q, (dq + ddq / 2. * dt) * dt);
    dq += ddq * dt;

    // Update data handler
    data_handler.updateInternalData(q, dq, true);

    // Check common to all tests
    check_joint_limits();
  }

  bool is_error_decreasing(std::string name, double error)
  {
    if (errors.count(name) == 0)
    {
      errors.insert({name, error});
      return true; // no further check
    }
    const bool res{error <= errors.at(name)};
    errors.at(name) = error; // Update value
    return res;
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
  CentroidalID::Settings settings;
  double dt = 1e-3;
  CentroidalID solver;

  double t = 0.;
  int step_i = 0;
  Eigen::VectorXd q;
  Eigen::VectorXd dq;
  Eigen::VectorXd ddq;
  Eigen::VectorXd tau;

  std::map<std::string, double> errors;
};

BOOST_AUTO_TEST_CASE(CentroidalID_postureTask)
{
  CentroidalID::Settings settings;
  settings.kp_posture = 20.;
  settings.w_posture = 1.;

  TestCentroidalID test(getSoloHandler(), settings);

  // Easy access
  const RobotModelHandler & model_handler = test.model_handler;
  const size_t nq = model_handler.getModel().nq;
  const size_t nv = model_handler.getModel().nv;

  // Target state
  const Eigen::VectorXd q_target =
    model_handler.getReferenceState().head(nq); // CentroidalID set posture target to reference configuration
  CentroidalID::FeetPoseVector feet_pose_vec;
  CentroidalID::FeetVelocityVector feet_vel_vec;
  for (int foot_nb; foot_nb < test.model_handler.getFeetNb(); foot_nb++)
  {
    feet_pose_vec.push_back(pinocchio::SE3::Identity());
    feet_vel_vec.push_back(pinocchio::Motion::Zero());
  }
  test.solver.setTarget(
    Eigen::VectorXd::Zero(3), Eigen::VectorXd::Zero(3), feet_pose_vec, feet_vel_vec, {false, false, false, false}, {});

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
    BOOST_CHECK(test.is_error_decreasing("posture", error));
  }
}

void test_contact(TestCentroidalID test)
{
  // Easy access
  const RobotModelHandler & model_handler = test.model_handler;
  const RobotDataHandler & data_handler = test.data_handler;
  const size_t nq = model_handler.getModel().nq;
  const size_t nv = model_handler.getModel().nv;

  // No need to set target as CentroidalID sets it by default to reference state
  const Eigen::VectorXd q_target = model_handler.getReferenceState().head(nq);

  // Let the robot stabilize
  const int N_STEP = 500;
  while (test.step_i < N_STEP)
  {
    // Solve
    test.step();

    // Check that contact velocity is null
    for (int foot_nb = 0; foot_nb < model_handler.getFeetNb(); foot_nb++)
    {
      const pinocchio::Motion foot_vel = pinocchio::getFrameVelocity(
        model_handler.getModel(), data_handler.getData(), model_handler.getFootFrameId(foot_nb), pinocchio::WORLD);
      BOOST_CHECK_LE(foot_vel.linear().norm(), 1e-2);
      if (model_handler.getFootType(foot_nb) == RobotModelHandler::FootType::QUAD)
      {
        // Rotation should also be null for quadrilateral contacts
        BOOST_CHECK_LE(foot_vel.angular().norm(), 1e-1);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(CentroidalID_contactPoint_cost)
{
  CentroidalID::Settings settings;
  settings.kp_base = 1.0;
  settings.kp_contact = 10.0;
  settings.w_base = 1.;
  settings.w_contact_motion = 10.0;
  settings.w_contact_force = 1.0;

  TestCentroidalID simu(getSoloHandler(), settings);
  simu.q = solo_q_start(simu.model_handler); // Set initial configuration
  test_contact(simu);
}

BOOST_AUTO_TEST_CASE(CentroidalID_contactQuad_cost)
{
  CentroidalID::Settings settings;
  settings.kp_base = 1.0;
  settings.kp_posture = 1.;
  settings.kp_contact = 10.0;
  settings.w_base = 1.;
  settings.w_posture = 0.05;
  settings.w_contact_motion = 10.0;
  settings.w_contact_force = 1.0;

  TestCentroidalID simu(getTalosModelHandler(), settings);
  test_contact(simu);
}

BOOST_AUTO_TEST_CASE(CentroidalID_contactPoint_equality)
{
  CentroidalID::Settings settings;
  settings.kp_base = 1.0;
  settings.kp_contact = 10.0;
  settings.w_base = 1.;
  settings.w_contact_motion = 10.0;
  settings.w_contact_force = 1.0;
  settings.contact_motion_equality = true;
  TestCentroidalID simu(getSoloHandler(), settings);
  simu.q = solo_q_start(simu.model_handler); // Set initial configuration
  test_contact(simu);
}

BOOST_AUTO_TEST_CASE(CentroidalID_contactQuad_equality)
{
  CentroidalID::Settings settings;
  settings.kp_base = 1.0;
  settings.kp_posture = 1.;
  settings.kp_contact = 10.0;
  settings.w_base = 1.;
  settings.w_posture = 0.05;
  settings.w_contact_motion = 10.0;
  settings.w_contact_force = 1.0;
  settings.contact_motion_equality = true;

  TestCentroidalID simu(getTalosModelHandler(), settings);
  test_contact(simu);
}

BOOST_AUTO_TEST_CASE(CentroidalID_baseTask)
{
  CentroidalID::Settings settings;
  settings.kp_base = 7.;
  settings.kp_contact = .1;
  settings.w_base = 100.0;
  settings.w_contact_force = 1.0;
  settings.w_contact_motion = 1.0;

  TestCentroidalID test(getSoloHandler(), settings);

  // Easy access
  const RobotModelHandler & model_handler = test.model_handler;
  const size_t nq = model_handler.getModel().nq;

  // No need to set target as CentroidalID sets it by default to reference state
  const Eigen::VectorXd q_target = model_handler.getReferenceState().head(nq);

  // Change initial state
  test.q = solo_q_start(model_handler);

  const int N_STEP = 10000;
  for (int i = 0; i < N_STEP; i++)
  {
    // Solve
    test.step();

    // Compute error
    const Eigen::VectorXd delta_pose = pinocchio::difference(model_handler.getModel(), test.q, q_target).head<6>();
    const double error = delta_pose.norm();

    // Checks
    if (error > 2e-2) // If haven't converged yet, should be strictly decreasing
      BOOST_CHECK(test.is_error_decreasing("base", error));
    if (i > 9 * N_STEP / 10) // Should have converged by now
      BOOST_CHECK(error < 2e-2);
  }
}

BOOST_AUTO_TEST_CASE(CentroidalID_allTasks)
{
  CentroidalID::Settings settings;
  settings.kp_base = 10.;
  settings.kp_posture = 1.;
  settings.kp_contact = 10.;
  settings.w_base = 10.0;
  settings.w_posture = 0.1;
  settings.w_contact_force = 1.0;
  settings.w_contact_motion = 1.0;

  TestCentroidalID test(getSoloHandler(), settings);

  // Easy access
  const RobotModelHandler & model_handler = test.model_handler;
  const size_t nq = model_handler.getModel().nq;
  const size_t nv = model_handler.getModel().nv;

  // No need to set target as CentroidalID sets it by default to reference state
  const Eigen::VectorXd q_target = model_handler.getReferenceState().head(nq);

  test.q = solo_q_start(model_handler);
  const int N_STEP = 1000;
  for (int i = 0; i < N_STEP; i++)
  {
    // Solve
    test.step();

    // Check error is decreasing
    const Eigen::VectorXd delta_q = pinocchio::difference(model_handler.getModel(), test.q, q_target);
    const double error = delta_q.norm();

    BOOST_CHECK(test.is_error_decreasing("q", error));
  }
}

BOOST_AUTO_TEST_SUITE_END()
