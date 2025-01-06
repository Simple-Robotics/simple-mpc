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
  pinocchio::Model modelFull;
  std::string urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_reduced.urdf";
  std::string srdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf";

  pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), modelFull);
  pinocchio::srdf::loadReferenceConfigurations(modelFull, srdf_path, false);
  pinocchio::srdf::loadRotorParameters(modelFull, srdf_path, false);

  // Lock joint list
  const std::vector<std::string> controlled_joints_names{
    "universe",          "root_joint",        "leg_left_1_joint",  "leg_left_2_joint",  "leg_left_3_joint",
    "leg_left_4_joint",  "leg_left_5_joint",  "leg_left_6_joint",  "leg_right_1_joint", "leg_right_2_joint",
    "leg_right_3_joint", "leg_right_4_joint", "leg_right_5_joint", "leg_right_6_joint", "torso_1_joint",
    "torso_2_joint",     "arm_left_1_joint",  "arm_left_2_joint",  "arm_left_3_joint",  "arm_left_4_joint",
    "arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint", "arm_right_4_joint",
  };

  std::vector<std::string> locked_joints_names{modelFull.names};
  std::vector<pinocchio::JointIndex> controlled_joints_ids;
  locked_joints_names.erase(
    std::remove_if(
      locked_joints_names.begin(), locked_joints_names.end(),
      [&controlled_joints_names](const std::string & name) {
        return std::find(controlled_joints_names.begin(), controlled_joints_names.end(), name)
               != controlled_joints_names.end();
      }),
    locked_joints_names.end());

  // Construct controlled and locked joints ids list
  pinocchio::Model model;
  std::vector<unsigned long> locked_joint_ids;
  for (size_t i = 1; i < modelFull.names.size(); i++)
  {
    const std::string joint_name = modelFull.names.at(i);
    if (count(locked_joints_names.begin(), locked_joints_names.end(), joint_name) == 0)
    {
      controlled_joints_ids.push_back(modelFull.getJointId(joint_name));
    }
    else
    {
      locked_joint_ids.push_back(modelFull.getJointId(joint_name));
    }
  }

  // Build reduced model with locked joints
  pinocchio::buildReducedModel(modelFull, locked_joint_ids, modelFull.referenceConfigurations["half_sitting"], model);

  // Actually create handler
  std::string base_joint = "root_joint";
  RobotModelHandler model_handler(model, "half_sitting", base_joint);

  // Add feet
  model_handler.addFoot(
    "left_sole_link", base_joint, pinocchio::SE3(Eigen::Quaternion(0., 0., 0., 1.), Eigen::Vector3d(0., 0.1, 0.)));
  model_handler.addFoot(
    "right_sole_link", base_joint, pinocchio::SE3(Eigen::Quaternion(0., 0., 0., 1.), Eigen::Vector3d(0., -0.1, 0.)));

  RobotDataHandler data_handler(model_handler);

  size_t T = 100;

  FullDynamicsSettings problem_settings;
  int nu = model_handler.getModel().nv - 6;
  int ndx = model_handler.getModel().nv * 2;

  Eigen::VectorXd w_x_vec(ndx);
  w_x_vec << 0, 0, 0, 100, 100, 100, // Base pos/ori
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1,    // Left leg
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1,    // Right leg
    10, 10,                          // Torso
    1, 1, 1, 1,                      // Left arm
    1, 1, 1, 1,                      // Right arm
    1, 1, 1, 1, 1, 1,                // Base pos/ori vel
    0.1, 0.1, 0.1, 0.1, 0.01, 0.01,  // Left leg vel
    0.1, 0.1, 0.1, 0.1, 0.01, 0.01,  // Right leg vel
    10, 10,                          // Torso vel
    1, 1, 1, 1,                      // Left arm vel
    1, 1, 1, 1;                      // Right arm vel
  Eigen::VectorXd w_cent(6);
  w_cent << 0, 0, 0, 0.1, 0., 100;
  Eigen::VectorXd w_forces(6);
  w_forces << 0.0001, 0.0001, 0.0001, 0.01, 0.01, 0.01;

  Eigen::VectorXd u0 = Eigen::VectorXd::Zero(nu);

  problem_settings.timestep = 0.01;
  problem_settings.w_x = Eigen::MatrixXd::Zero(ndx, ndx);
  problem_settings.w_x.diagonal() = w_x_vec;
  problem_settings.w_u = Eigen::MatrixXd::Identity(nu, nu) * 1e-4;
  problem_settings.w_cent = Eigen::MatrixXd::Zero(6, 6);
  problem_settings.w_cent.diagonal() = w_cent;
  problem_settings.gravity = {0, 0, -9.81};
  problem_settings.force_size = 6;
  problem_settings.Kp_correction = Eigen::VectorXd::Ones(6);
  problem_settings.Kd_correction = Eigen::VectorXd::Ones(6);
  problem_settings.w_forces = Eigen::MatrixXd::Zero(6, 6);
  problem_settings.w_forces.diagonal() = w_forces;
  problem_settings.w_frame = Eigen::MatrixXd::Identity(6, 6) * 2000;
  problem_settings.umin = -model_handler.getModel().effortLimit.tail(nu);
  problem_settings.umax = model_handler.getModel().effortLimit.tail(nu);
  problem_settings.qmin = model_handler.getModel().lowerPositionLimit.tail(nu);
  problem_settings.qmax = model_handler.getModel().upperPositionLimit.tail(nu);
  problem_settings.mu = 0.8;
  problem_settings.Lfoot = 0.1;
  problem_settings.Wfoot = 0.075;

  std::shared_ptr<OCPHandler> ocpPtr = std::make_shared<FullDynamicsOCP>(problem_settings, model_handler, data_handler);
  ocpPtr->createProblem(model_handler.getReferenceState(), T, 6, problem_settings.gravity[2], true);

  MPCSettings mpc_settings;
  mpc_settings.support_force = -problem_settings.gravity[2] * model_handler.getMass();
  mpc_settings.TOL = 1e-4;
  mpc_settings.mu_init = 1e-8;
  mpc_settings.max_iters = 1;
  mpc_settings.num_threads = 2;

  MPC mpc{mpc_settings, ocpPtr};

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
  for (std::size_t i = 0; i < 10; i++)
  {
    std::map<std::string, bool> contact_state;
    contact_state.insert({model_handler.getFootName(0), true});
    contact_state.insert({model_handler.getFootName(1), true});
    contact_states.push_back(contact_state);
  }

  mpc.generateCycleHorizon(contact_states);

  return 0;
}
