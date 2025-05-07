
#include <aligator/core/manifold-base.hpp>
#include <boost/test/unit_test.hpp>

#include "simple-mpc/qp-solvers.hpp"
#include "simple-mpc/robot-handler.hpp"
#include "test_utils.cpp"

BOOST_AUTO_TEST_SUITE(qpsolvers)

using namespace simple_mpc;

BOOST_AUTO_TEST_CASE(ID_solver)
{
  RobotModelHandler model_handler = getTalosModelHandler();
  RobotDataHandler data_handler(model_handler);

  IDSettings settings;
  settings.contact_ids = model_handler.getFeetIds();
  settings.mu = 0.8;
  settings.Lfoot = 0.1;
  settings.Wfoot = 0.075;
  settings.force_size = 6;
  settings.kd = 10;
  settings.w_force = 1000;
  settings.w_acc = 1;
  settings.w_tau = 0;
  settings.verbose = false;

  IDSolver ID_solver(settings, model_handler.getModel());

  std::vector<bool> contact_states;
  contact_states.push_back(true);
  contact_states.push_back(true);

  Eigen::VectorXd v = Eigen::VectorXd::Random(model_handler.getModel().nv);
  Eigen::VectorXd a = Eigen::VectorXd::Random(model_handler.getModel().nv);
  Eigen::VectorXd tau = Eigen::VectorXd::Zero(model_handler.getModel().nv - 6);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(6 * 2);

  Eigen::MatrixXd M = data_handler.getData().M;
  pinocchio::Data rdata = data_handler.getData();
  ID_solver.solveQP(rdata, contact_states, v, a, tau, forces, M);
}

BOOST_AUTO_TEST_CASE(IKID_solver)
{
  RobotModelHandler model_handler = getTalosModelHandler();
  RobotDataHandler data_handler(model_handler);

  std::vector<FrameIndex> vec_base;
  std::vector<Eigen::VectorXd> Kp;
  std::vector<Eigen::VectorXd> Kd;

  Eigen::VectorXd g_q(model_handler.getModel().nv);
  g_q << 0, 0, 0, 100, 100, 100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10, 100, 100, 100, 100, 100, 100, 100, 100;

  double g_p = 400;
  double g_b = 10;

  Kp.push_back(g_q);
  Kp.push_back(Eigen::VectorXd::Constant(6, g_p));
  Kp.push_back(Eigen::VectorXd::Constant(3, g_b));

  Kd.push_back(2 * g_q.array().sqrt());
  Kd.push_back(Eigen::VectorXd::Constant(6, 2 * sqrt(g_p)));
  Kd.push_back(Eigen::VectorXd::Constant(3, 2 * sqrt(g_b)));

  vec_base.push_back(model_handler.getBaseFrameId());
  IKIDSettings settings;
  settings.contact_ids = model_handler.getFeetIds();
  settings.fixed_frame_ids = vec_base;
  settings.x0 = model_handler.getReferenceState();
  settings.Kp_gains = Kp, settings.Kd_gains = Kd, settings.dt = 0.01, settings.mu = 0.8;
  settings.Lfoot = 0.1;
  settings.Wfoot = 0.075;
  settings.force_size = 6;
  settings.w_qref = 500;
  settings.w_footpose = 50000;
  settings.w_centroidal = 10;
  settings.w_centroidal = 10;
  settings.w_baserot = 1000;
  settings.w_force = 100;
  settings.verbose = false;

  IKIDSolver IKID_solver(settings, model_handler.getModel());

  std::vector<bool> contact_states;
  contact_states.push_back(true);
  contact_states.push_back(true);

  std::vector<pinocchio::SE3> foot_refs;
  std::vector<pinocchio::SE3> foot_refs_next;
  for (auto const & name : model_handler.getFeetNames())
  {
    pinocchio::SE3 foot_ref = data_handler.getFootPose(name);
    foot_refs.push_back(foot_ref);
    foot_ref.translation()[0] += 0.1;

    foot_refs_next.push_back(foot_ref);
  }

  Eigen::VectorXd dq = Eigen::VectorXd::Random(model_handler.getModel().nv);
  Eigen::VectorXd dv = Eigen::VectorXd::Random(model_handler.getModel().nv);
  Eigen::VectorXd dH = Eigen::VectorXd::Random(6);
  Eigen::VectorXd x0 = model_handler.getReferenceState();
  Eigen::VectorXd xm = model_handler.getReferenceState();
  Eigen::VectorXd new_q = pinocchio::integrate(model_handler.getModel(), x0.head(model_handler.getModel().nq), dq);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(6 * 2);
  xm << new_q, dv;

  Eigen::MatrixXd M = data_handler.getData().M;
  pinocchio::Data rdata = data_handler.getData();

  IKID_solver.computeDifferences(rdata, xm, foot_refs, foot_refs_next);
  IKID_solver.solve_qp(rdata, contact_states, dv, forces, dH, M);
}

BOOST_AUTO_TEST_SUITE_END()
