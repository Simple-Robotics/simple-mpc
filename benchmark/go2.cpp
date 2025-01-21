#include "simple-mpc/fulldynamics.hpp"
#include "simple-mpc/mpc.hpp"
#include "simple-mpc/ocp-handler.hpp"
#include "simple-mpc/robot-handler.hpp"
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>

using simple_mpc::ContactMap;
using simple_mpc::RobotDataHandler;
using simple_mpc::RobotModelHandler;
using PoseVec = aligator::StdVectorEigenAligned<Eigen::Vector3d>;
using simple_mpc::FullDynamicsOCP;
using simple_mpc::FullDynamicsSettings;
using simple_mpc::MPC;
using simple_mpc::MPCSettings;
using simple_mpc::OCPHandler;

int main()
{
  // Load pinocchio model from example robot data
  pinocchio::Model model;
  std::string urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/urdf/go2.urdf";
  std::string srdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/srdf/go2.srdf";
  std::string base_joint_name = "root_joint";

  pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model);
  pinocchio::srdf::loadReferenceConfigurations(model, srdf_path, false);
  // pinocchio::srdf::loadRotorParameters(model, srdf_path, false);

  simple_mpc::RobotModelHandler model_handler = simple_mpc::RobotModelHandler(model, "standing", base_joint_name);

  /// Add reference foot for walking
  pinocchio::SE3 ref_FL_foot = pinocchio::SE3::Identity();
  pinocchio::SE3 ref_FR_foot = pinocchio::SE3::Identity();
  pinocchio::SE3 ref_RL_foot = pinocchio::SE3::Identity();
  pinocchio::SE3 ref_RR_foot = pinocchio::SE3::Identity();
  ref_FL_foot.translation() = Eigen::Vector3d(0.17, 0.15, 0.0);
  ref_FR_foot.translation() = Eigen::Vector3d(0.17, -0.15, 0.0);
  ref_RL_foot.translation() = Eigen::Vector3d(-0.24, 0.15, 0.0);
  ref_RR_foot.translation() = Eigen::Vector3d(-0.24, -0.15, 0.0);
  model_handler.addFoot("FL_foot", base_joint_name, ref_FL_foot);
  model_handler.addFoot("FR_foot", base_joint_name, ref_FR_foot);
  model_handler.addFoot("RL_foot", base_joint_name, ref_RL_foot);
  model_handler.addFoot("RR_foot", base_joint_name, ref_RR_foot);

  RobotDataHandler data_handler(model_handler);

  int nu = model_handler.getModel().nv - 6;
  int nv = model_handler.getModel().nv;
  int ndx = nv * 2;
  Eigen::Vector3d gravity;
  gravity << 0., 0., -9.81;

  // Define settings for OCP
  simple_mpc::FullDynamicsSettings problem_settings;

  Eigen::VectorXd w_x_vec(ndx);
  w_x_vec << 0, 0, 0, 0., 0., 0,  // Base pos/ori
    10., 10., 10., 10., 10., 10., // FL FR leg
    10., 10., 10., 10., 10., 10., // RL RR leg
    10., 10., 10., 10., 10., 10., // Base vel
    .1, .1, .1, .1, .1, .1,       // FL FR vel
    .1, .1, .1, .1, .1, .1;       // RL RR vel
  Eigen::VectorXd w_cent(6);
  w_cent << 0, 0, 0, 0, 0, 0;
  Eigen::VectorXd w_forces(3);
  w_forces << 0.0001, 0.0001, 0.0001;

  Eigen::VectorXd u0 = Eigen::VectorXd::Zero(nu);

  problem_settings.timestep = 0.01;
  problem_settings.w_x = Eigen::MatrixXd::Zero(ndx, ndx);
  problem_settings.w_x.diagonal() = w_x_vec;
  problem_settings.w_u = Eigen::MatrixXd::Identity(nu, nu) * 1e-4;
  problem_settings.w_cent = Eigen::MatrixXd::Zero(6, 6);
  problem_settings.w_cent.diagonal() = w_cent;
  problem_settings.gravity = gravity;
  problem_settings.force_size = 3;
  problem_settings.Kp_correction = Eigen::VectorXd::Zero(3);
  problem_settings.Kd_correction = Eigen::VectorXd::Zero(3);
  problem_settings.w_forces = Eigen::MatrixXd::Zero(3, 3);
  problem_settings.w_forces.diagonal() = w_forces;
  problem_settings.w_frame = Eigen::MatrixXd::Identity(3, 3) * 1000;
  problem_settings.umin = -model_handler.getModel().effortLimit.tail(nu);
  problem_settings.umax = model_handler.getModel().effortLimit.tail(nu);
  problem_settings.qmin = model_handler.getModel().lowerPositionLimit.tail(nu);
  problem_settings.qmax = model_handler.getModel().upperPositionLimit.tail(nu);
  problem_settings.mu = 0.8;
  problem_settings.Lfoot = 0.01;
  problem_settings.Wfoot = 0.01;
  problem_settings.torque_limits = false;
  problem_settings.kinematics_limits = false;
  problem_settings.force_cone = false;

  std::shared_ptr<simple_mpc::OCPHandler> ocpPtr =
    std::make_shared<simple_mpc::FullDynamicsOCP>(problem_settings, model_handler);

  // Create an OCP problem with horizon size T
  size_t T = 50;
  ocpPtr->createProblem(model_handler.getReferenceState(), T, 3, gravity[2], false);

  // Define settings for MPC
  int T_fly = 30;
  int T_contact = 5;
  simple_mpc::MPCSettings mpc_settings;
  mpc_settings.ddpIteration = 1;
  mpc_settings.support_force = -gravity[2] * model_handler.getMass();
  mpc_settings.TOL = 1e-4;
  mpc_settings.mu_init = 1e-8;
  mpc_settings.max_iters = 1;
  mpc_settings.num_threads = 1;
  mpc_settings.swing_apex = 0.2;
  mpc_settings.T_fly = T_fly;
  mpc_settings.T_contact = T_contact;
  mpc_settings.timestep = 0.01;

  std::shared_ptr<simple_mpc::MPC> mpc = std::make_shared<simple_mpc::MPC>(mpc_settings, ocpPtr);

  // Define the walking gait
  std::vector<std::map<std::string, bool>> contact_states;
  std::map<std::string, bool> contact_state_quadru;
  std::map<std::string, bool> contact_phase_lift_FL_RR;
  std::map<std::string, bool> contact_phase_lift_RL_FR;

  contact_state_quadru.insert({"FL_foot", true});
  contact_state_quadru.insert({"FR_foot", true});
  contact_state_quadru.insert({"RL_foot", true});
  contact_state_quadru.insert({"RR_foot", true});

  contact_phase_lift_FL_RR.insert({"FL_foot", false});
  contact_phase_lift_FL_RR.insert({"FR_foot", true});
  contact_phase_lift_FL_RR.insert({"RL_foot", true});
  contact_phase_lift_FL_RR.insert({"RR_foot", false});

  contact_phase_lift_RL_FR.insert({"FL_foot", true});
  contact_phase_lift_RL_FR.insert({"FR_foot", false});
  contact_phase_lift_RL_FR.insert({"RL_foot", false});
  contact_phase_lift_RL_FR.insert({"RR_foot", true});

  for (int i = 0; i < T_contact; i++)
    contact_states.push_back(contact_state_quadru);
  for (int i = 0; i < T_fly; i++)
    contact_states.push_back(contact_phase_lift_FL_RR);
  for (int i = 0; i < T_contact; i++)
    contact_states.push_back(contact_state_quadru);
  for (int i = 0; i < T_fly; i++)
    contact_states.push_back(contact_phase_lift_RL_FR);

  mpc->generateCycleHorizon(contact_states);

  return 0;
}
