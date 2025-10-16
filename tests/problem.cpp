#include <boost/test/unit_test.hpp>

#include "simple-mpc/arm-dynamics.hpp"
#include "simple-mpc/centroidal-dynamics.hpp"
#include "simple-mpc/fulldynamics.hpp"
#include "simple-mpc/fwd.hpp"
#include "simple-mpc/kinodynamics.hpp"
#include "simple-mpc/ocp-handler.hpp"
#include "simple-mpc/robot-handler.hpp"
#include "test_utils.cpp"

BOOST_AUTO_TEST_SUITE(problem)

using namespace simple_mpc;

BOOST_AUTO_TEST_CASE(fulldynamics)
{
  RobotModelHandler model_handler = getTalosModelHandler();
  RobotDataHandler data_handler(model_handler);

  std::vector<std::string> contact_names = {"left_sole_link", "right_sole_link"};
  std::map<std::string, bool> contact_states;
  std::map<std::string, bool> land_constraint;
  std::map<std::string, pinocchio::SE3> contact_poses;
  contact_states.insert({contact_names[0], true});
  contact_states.insert({contact_names[1], false});

  land_constraint.insert({contact_names[0], true});
  land_constraint.insert({contact_names[1], false});

  pinocchio::SE3 p1 = data_handler.getFootPose(0);
  pinocchio::SE3 p2 = data_handler.getFootPose(1);
  p1.translation() << 0, 0.1, 0;
  p2.translation() << 0, -0.1, 0;
  contact_poses.insert({contact_names[0], p1});
  contact_poses.insert({contact_names[1], p2});

  std::map<std::string, Eigen::VectorXd> force_refs;
  Eigen::VectorXd f1(6);
  f1 << 0, 0, 800, 0, 0, 0;
  force_refs.insert({"left_sole_link", f1});
  force_refs.insert({"right_sole_link", Eigen::VectorXd::Zero(6)});

  FullDynamicsSettings settings = getFullDynamicsSettings(model_handler);
  FullDynamicsOCP fdproblem(settings, model_handler);
  StageModel sm = fdproblem.createStage(contact_states, contact_poses, force_refs, land_constraint);
  CostStack * cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 6);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 3);

  fdproblem.createProblem(model_handler.getReferenceState(), 100, 6, settings.gravity[2], true);

  CostStack * csp = dynamic_cast<CostStack *>(&*fdproblem.getProblem().stages_[0]->cost_);
  QuadraticControlCost * cc = csp->getComponent<QuadraticControlCost>("control_cost");
  QuadraticResidualCost * crc = csp->getComponent<QuadraticResidualCost>("centroidal_cost");
  QuadraticResidualCost * cpc = csp->getComponent<QuadraticResidualCost>("left_sole_link_pose_cost");

  std::vector<bool> cs2 = {true, true};
  BOOST_CHECK_EQUAL(fdproblem.getSize(), 100);
  BOOST_CHECK_EQUAL(fdproblem.getContactSupport(2), 2);
  BOOST_TEST(fdproblem.getContactState(2) == cs2);
  BOOST_CHECK_EQUAL(cc->weights_, settings.w_u);
  BOOST_CHECK_EQUAL(crc->weights_, settings.w_cent);
  BOOST_CHECK_EQUAL(cpc->weights_, settings.w_frame);

  pinocchio::SE3 pose_left_random = pinocchio::SE3::Random();
  fdproblem.setReferencePose(4, "left_sole_link", pose_left_random);

  BOOST_CHECK_EQUAL(fdproblem.getReferencePose(4, "left_sole_link"), pose_left_random);

  pinocchio::SE3 new_pose_left = pinocchio::SE3::Identity();
  new_pose_left.translation() << 1, 0, 2;
  pinocchio::SE3 new_pose_right = pinocchio::SE3::Identity();
  new_pose_right.translation() << -1, 0, 2;
  std::map<std::string, pinocchio::SE3> new_poses;
  new_poses.insert({"left_sole_link", new_pose_left});
  new_poses.insert({"right_sole_link", new_pose_right});

  fdproblem.setReferencePoses(3, new_poses);
  BOOST_CHECK_EQUAL(fdproblem.getReferencePose(3, "left_sole_link"), new_poses.at("left_sole_link"));
  BOOST_CHECK_EQUAL(fdproblem.getReferencePose(3, "right_sole_link"), new_poses.at("right_sole_link"));

  force_refs.at("left_sole_link")[1] = 1;
  force_refs.at("right_sole_link")[0] = 1;
  fdproblem.setReferenceForces(3, force_refs);
  BOOST_CHECK_EQUAL(fdproblem.getReferenceForce(3, "left_sole_link"), force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(fdproblem.getReferenceForce(3, "right_sole_link"), force_refs.at("right_sole_link"));

  force_refs.at("left_sole_link")[2] = 250;
  fdproblem.setReferenceForce(5, "left_sole_link", force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(fdproblem.getReferenceForce(5, "left_sole_link"), force_refs.at("left_sole_link"));

  Eigen::VectorXd pose_base(7);
  pose_base << 0, 0, 2, 0, 0, 0, 1;
  fdproblem.setPoseBase(2, pose_base);
  BOOST_CHECK_EQUAL(fdproblem.getPoseBase(2), pose_base);

  Eigen::VectorXd new_q = pinocchio::randomConfiguration(model_handler.getModel());
  Eigen::VectorXd new_x(model_handler.getModel().nq + model_handler.getModel().nv);
  new_x.head(model_handler.getModel().nq) = new_q;
  new_x.tail(model_handler.getModel().nv).setRandom();
  fdproblem.setReferenceState(2, new_x);
  BOOST_CHECK_EQUAL(fdproblem.getReferenceState(2), new_x);
}

BOOST_AUTO_TEST_CASE(kinodynamics)
{
  RobotModelHandler model_handler = getTalosModelHandler();
  RobotDataHandler data_handler(model_handler);

  std::vector<std::string> contact_names = {"left_sole_link", "right_sole_link"};
  std::map<std::string, bool> contact_states;
  std::map<std::string, bool> land_constraint;
  std::map<std::string, pinocchio::SE3> contact_poses;
  contact_states.insert({contact_names[0], true});
  contact_states.insert({contact_names[1], false});

  land_constraint.insert({contact_names[0], true});
  land_constraint.insert({contact_names[1], false});

  pinocchio::SE3 p1 = data_handler.getFootPose(0);
  pinocchio::SE3 p2 = data_handler.getFootPose(1);
  p1.translation() << 0, 0.1, 0;
  p2.translation() << 0, -0.1, 0;
  contact_poses.insert({contact_names[0], p1});
  contact_poses.insert({contact_names[1], p2});

  KinodynamicsSettings settings = getKinodynamicsSettings(model_handler);
  KinodynamicsOCP knproblem(settings, model_handler);

  std::map<std::string, Eigen::VectorXd> force_refs;
  Eigen::VectorXd f1(6);
  f1 << 0, 0, 800, 0, 0, 0;
  force_refs.insert({"left_sole_link", f1});
  force_refs.insert({"right_sole_link", Eigen::VectorXd::Zero(6)});
  StageModel sm = knproblem.createStage(contact_states, contact_poses, force_refs, land_constraint);
  CostStack * cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 6);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 3);

  knproblem.createProblem(model_handler.getReferenceState(), 100, 6, settings.gravity[2], true);

  CostStack * csp = dynamic_cast<CostStack *>(&*knproblem.getProblem().stages_[0]->cost_);
  QuadraticControlCost * cc = csp->getComponent<QuadraticControlCost>("control_cost");
  QuadraticResidualCost * crc = csp->getComponent<QuadraticResidualCost>("centroidal_cost");
  QuadraticResidualCost * cpc = csp->getComponent<QuadraticResidualCost>("left_sole_link_pose_cost");

  std::vector<bool> cs2 = {true, true};
  BOOST_CHECK_EQUAL(knproblem.getContactSupport(2), 2);
  BOOST_TEST(knproblem.getContactState(2) == cs2);
  BOOST_CHECK_EQUAL(cc->weights_, settings.w_u);
  BOOST_CHECK_EQUAL(crc->weights_, settings.w_cent);
  BOOST_CHECK_EQUAL(cpc->weights_, settings.w_frame);
  BOOST_CHECK_EQUAL(knproblem.getSize(), 100);

  pinocchio::SE3 pose_left_random = pinocchio::SE3::Random();
  knproblem.setReferencePose(4, "left_sole_link", pose_left_random);

  BOOST_CHECK_EQUAL(knproblem.getReferencePose(4, "left_sole_link"), pose_left_random);

  pinocchio::SE3 new_pose_left = pinocchio::SE3::Identity();
  new_pose_left.translation() << 1, 0, 2;
  pinocchio::SE3 new_pose_right = pinocchio::SE3::Identity();
  new_pose_right.translation() << -1, 0, 2;
  std::map<std::string, pinocchio::SE3> new_poses;
  new_poses.insert({"left_sole_link", new_pose_left});
  new_poses.insert({"right_sole_link", new_pose_right});

  knproblem.setReferencePoses(3, new_poses);

  BOOST_CHECK_EQUAL(knproblem.getReferencePose(3, "left_sole_link"), new_poses.at("left_sole_link"));
  BOOST_CHECK_EQUAL(knproblem.getReferencePose(3, "right_sole_link"), new_poses.at("right_sole_link"));

  force_refs.at("left_sole_link")[1] = 1;
  force_refs.at("right_sole_link")[0] = 1;
  knproblem.setReferenceForces(3, force_refs);
  BOOST_CHECK_EQUAL(knproblem.getReferenceForce(3, "left_sole_link"), force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(knproblem.getReferenceForce(3, "right_sole_link"), force_refs.at("right_sole_link"));

  force_refs.at("left_sole_link")[2] = 250;
  knproblem.setReferenceForce(5, "left_sole_link", force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(knproblem.getReferenceForce(5, "left_sole_link"), force_refs.at("left_sole_link"));

  Eigen::VectorXd pose_base(7);
  pose_base << 0, 0, 2, 0, 0, 0, 1;
  knproblem.setPoseBase(2, pose_base);
  BOOST_CHECK_EQUAL(knproblem.getPoseBase(2), pose_base);

  Eigen::VectorXd new_q = pinocchio::randomConfiguration(model_handler.getModel());
  Eigen::VectorXd new_x(model_handler.getModel().nq + model_handler.getModel().nv);
  new_x.head(model_handler.getModel().nq) = new_q;
  new_x.tail(model_handler.getModel().nv).setRandom();
  knproblem.setReferenceState(2, new_x);
  BOOST_CHECK_EQUAL(knproblem.getReferenceState(2), new_x);
}

BOOST_AUTO_TEST_CASE(centroidal)
{
  RobotModelHandler model_handler = getTalosModelHandler();
  RobotDataHandler data_handler(model_handler);

  CentroidalSettings settings = getCentroidalSettings();
  CentroidalOCP cproblem(settings, model_handler);

  std::vector<std::string> contact_names = {"left_sole_link", "right_sole_link"};
  std::map<std::string, bool> contact_states;
  std::map<std::string, bool> land_constraint;
  std::map<std::string, pinocchio::SE3> contact_poses;
  contact_states.insert({contact_names[0], true});
  contact_states.insert({contact_names[1], false});

  land_constraint.insert({contact_names[0], true});
  land_constraint.insert({contact_names[1], false});

  pinocchio::SE3 p1 = data_handler.getFootPose(0);
  pinocchio::SE3 p2 = data_handler.getFootPose(1);
  p1.translation() << 0, 0.1, 0;
  p2.translation() << 0, -0.1, 0;
  contact_poses.insert({contact_names[0], p1});
  contact_poses.insert({contact_names[1], p2});

  std::map<std::string, Eigen::VectorXd> force_refs;
  Eigen::VectorXd f1(6);
  f1 << 0, 0, 800, 0, 0, 0;
  force_refs.insert({"left_sole_link", f1});
  force_refs.insert({"right_sole_link", Eigen::VectorXd::Zero(6)});
  StageModel sm = cproblem.createStage(contact_states, contact_poses, force_refs, land_constraint);
  CostStack * cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 6);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 1);

  cproblem.createProblem(data_handler.getCentroidalState(), 100, 6, settings.gravity[2], true);

  CostStack * csp = dynamic_cast<CostStack *>(&*cproblem.getProblem().stages_[0]->cost_);
  QuadraticControlCost * cc = csp->getComponent<QuadraticControlCost>("control_cost");
  QuadraticResidualCost * crc = csp->getComponent<QuadraticResidualCost>("linear_mom_cost");
  QuadraticResidualCost * cpc = csp->getComponent<QuadraticResidualCost>("angular_acc_cost");

  std::vector<bool> cs2 = {true, true};
  BOOST_CHECK_EQUAL(cproblem.getContactSupport(2), 2);
  BOOST_TEST(cproblem.getContactState(2) == cs2);
  BOOST_CHECK_EQUAL(cc->weights_, settings.w_u);
  BOOST_CHECK_EQUAL(crc->weights_, settings.w_linear_mom);
  BOOST_CHECK_EQUAL(cpc->weights_, settings.w_angular_acc);
  BOOST_CHECK_EQUAL(cproblem.getSize(), 100);

  force_refs.at("left_sole_link")[1] = 1;
  force_refs.at("right_sole_link")[0] = 1;
  cproblem.setReferenceForces(3, force_refs);
  BOOST_CHECK_EQUAL(cproblem.getReferenceForce(3, "left_sole_link"), force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(cproblem.getReferenceForce(3, "right_sole_link"), force_refs.at("right_sole_link"));

  force_refs.at("left_sole_link")[2] = 250;
  cproblem.setReferenceForce(5, "left_sole_link", force_refs.at("left_sole_link"));
  BOOST_CHECK_EQUAL(cproblem.getReferenceForce(5, "left_sole_link"), force_refs.at("left_sole_link"));

  pinocchio::SE3 pose_left_random = pinocchio::SE3::Random();
  cproblem.setReferencePose(4, "left_sole_link", pose_left_random);

  BOOST_CHECK_EQUAL(cproblem.getReferencePose(4, "left_sole_link").translation(), pose_left_random.translation());

  pinocchio::SE3 new_pose_left = pinocchio::SE3::Identity();
  new_pose_left.translation() << 1, 0, 2;
  pinocchio::SE3 new_pose_right = pinocchio::SE3::Identity();
  new_pose_right.translation() << -1, 0, 2;
  std::map<std::string, pinocchio::SE3> new_poses;
  new_poses.insert({"left_sole_link", new_pose_left});
  new_poses.insert({"right_sole_link", new_pose_right});

  cproblem.setReferencePoses(3, new_poses);

  BOOST_CHECK_EQUAL(cproblem.getReferencePose(3, "left_sole_link"), new_poses.at("left_sole_link"));
  BOOST_CHECK_EQUAL(cproblem.getReferencePose(3, "right_sole_link"), new_poses.at("right_sole_link"));

  Eigen::VectorXd pose_base(3);
  pose_base << 0, 0, 2;
  cproblem.setPoseBase(2, pose_base);
  BOOST_CHECK_EQUAL(cproblem.getPoseBase(2), pose_base);

  Eigen::VectorXd new_x(9);
  new_x << 0, 0, 1, 0, 0.1, 0.1, 0.2, 0.2, 0.2;
  cproblem.setReferenceState(2, new_x);
  BOOST_CHECK_EQUAL(cproblem.getReferenceState(2), new_x);
}

BOOST_AUTO_TEST_CASE(centroidal_solo)
{
  RobotModelHandler model_handler = getSoloHandler();
  RobotDataHandler data_handler(model_handler);

  CentroidalSettings settings = getCentroidalSettings();
  settings.force_size = 3;

  CentroidalOCP cproblem(settings, model_handler);

  std::vector<std::string> contact_names = {"FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"};
  std::map<std::string, bool> contact_states;
  std::map<std::string, pinocchio::SE3> contact_poses;

  contact_states.insert({contact_names[0], true});
  contact_states.insert({contact_names[1], true});
  contact_states.insert({contact_names[2], true});
  contact_states.insert({contact_names[3], false});
  pinocchio::SE3 p1 = data_handler.getFootPose(model_handler.getFootNb("FR_FOOT"));
  pinocchio::SE3 p2 = data_handler.getFootPose(model_handler.getFootNb("FL_FOOT"));
  pinocchio::SE3 p3 = data_handler.getFootPose(model_handler.getFootNb("HR_FOOT"));
  pinocchio::SE3 p4 = data_handler.getFootPose(model_handler.getFootNb("HL_FOOT"));

  contact_poses.insert({contact_names[0], p1});
  contact_poses.insert({contact_names[1], p2});
  contact_poses.insert({contact_names[2], p3});
  contact_poses.insert({contact_names[3], p4});

  std::map<std::string, Eigen::VectorXd> force_refs;
  Eigen::VectorXd f1(3);
  f1 << 0, 0, model_handler.getMass() / 3;
  force_refs.insert({"FR_FOOT", f1});
  force_refs.insert({"FL_FOOT", f1});
  force_refs.insert({"HR_FOOT", f1});
  force_refs.insert({"HL_FOOT", Eigen::VectorXd::Zero(3)});
  StageModel sm = cproblem.createStage(contact_states, contact_poses, force_refs, contact_states);
  CostStack * cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 6);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 3);

  cproblem.createProblem(data_handler.getCentroidalState(), 100, 3, settings.gravity[2], true);

  CostStack * csp = dynamic_cast<CostStack *>(&*cproblem.getProblem().stages_[0]->cost_);
  QuadraticControlCost * cc = csp->getComponent<QuadraticControlCost>("control_cost");
  QuadraticResidualCost * crc = csp->getComponent<QuadraticResidualCost>("linear_mom_cost");
  QuadraticResidualCost * cpc = csp->getComponent<QuadraticResidualCost>("angular_acc_cost");

  std::vector<bool> cs2 = {true, true, true, true};
  BOOST_CHECK_EQUAL(cproblem.getContactSupport(2), 4);
  BOOST_TEST(cproblem.getContactState(2) == cs2);
  BOOST_CHECK_EQUAL(cc->weights_, settings.w_u);
  BOOST_CHECK_EQUAL(crc->weights_, settings.w_linear_mom);
  BOOST_CHECK_EQUAL(cpc->weights_, settings.w_angular_acc);
  BOOST_CHECK_EQUAL(cproblem.getSize(), 100);

  force_refs.at("FR_FOOT")[1] = 1;
  force_refs.at("FL_FOOT")[0] = 1;
  cproblem.setReferenceForces(3, force_refs);
  BOOST_CHECK_EQUAL(cproblem.getReferenceForce(3, "FR_FOOT"), force_refs.at("FR_FOOT"));
  BOOST_CHECK_EQUAL(cproblem.getReferenceForce(3, "FL_FOOT"), force_refs.at("FL_FOOT"));
}

BOOST_AUTO_TEST_CASE(armdynamics)
{
  RobotModelHandler model_handler = getPandaHandler();
  RobotDataHandler data_handler(model_handler);

  ArmDynamicsSettings settings = getArmSettings(model_handler);
  ArmDynamicsOCP armproblem(settings, model_handler);
  Eigen::Vector3d reach_pose;
  reach_pose << 0.1, 0.1, 0.1;
  StageModel sm = armproblem.createStage(true, reach_pose);
  CostStack * cs = dynamic_cast<CostStack *>(&*sm.cost_);

  BOOST_CHECK_EQUAL(cs->components_.size(), 3);
  BOOST_CHECK_EQUAL(sm.numConstraints(), 2);

  armproblem.createProblem(model_handler.getReferenceState(), 100);

  CostStack * csp = dynamic_cast<CostStack *>(&*armproblem.getProblem().stages_[0]->cost_);
  QuadraticControlCost * cc = csp->getComponent<QuadraticControlCost>("control_cost");
  QuadraticResidualCost * crc = csp->getComponent<QuadraticResidualCost>(settings.ee_name + "_cost");

  std::vector<bool> cs2 = {true, true};
  BOOST_CHECK_EQUAL(armproblem.getSize(), 100);
  BOOST_CHECK_EQUAL(cc->weights_, settings.w_u);
  BOOST_CHECK_EQUAL(crc->weights_, settings.w_frame);

  Eigen::Vector3d pose_left_random;
  pose_left_random << 0.2, 0.3, 0.4;
  armproblem.setReferencePose(4, pose_left_random);

  BOOST_CHECK_EQUAL(armproblem.getReferencePose(4), pose_left_random);

  Eigen::VectorXd new_q = pinocchio::randomConfiguration(model_handler.getModel());
  Eigen::VectorXd new_x(model_handler.getModel().nq + model_handler.getModel().nv);
  new_x.head(model_handler.getModel().nq) = new_q;
  new_x.tail(model_handler.getModel().nv).setRandom();
  armproblem.setReferenceState(2, new_x);
  BOOST_CHECK_EQUAL(armproblem.getReferenceState(2), new_x);
}

BOOST_AUTO_TEST_SUITE_END()
