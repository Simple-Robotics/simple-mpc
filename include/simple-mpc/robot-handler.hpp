///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <pinocchio/fwd.hpp>
// Include pinocchio first
#include <Eigen/Dense>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <string>
#include <vector>

#include "simple-mpc/fwd.hpp"

namespace simple_mpc {
using namespace pinocchio;

/**
 * @brief Class managing every robot-related quantities.
 *
 * It holds the robot data, controlled joints, end-effector names
 * and other useful items.
 */
struct RobotModelHandler {
public:
  /**
   * @brief Robot model with all joints unlocked
   */
  Model model_full;

  /**
   * @brief Reduced model to be used by ocp
   */
  Model model;

  /**
   * @brief Robot total mass
   */
  double mass;

  /**
   * @brief Joint id to be controlled in full model
   */
  std::vector<unsigned long> controlled_joints_ids;

  /**
   * @brief Names of the frames to be in contact with the environment
   */
  std::vector<std::string> feet_names;

  /**
   * @brief Ids of the frames to be in contact with the environment
   */
  std::vector<FrameIndex> feet_ids;

  /**
   * @brief Ids of the frames that are reference position for the feet
   */
  std::vector<FrameIndex> ref_feet_ids;

  /**
   * @brief Name of the configuration to use as reference
   */
  Eigen::VectorXd reference_configuration;

  /**
   * @brief Root frame id
   */
  pinocchio::FrameIndex root_id;

public:
  /**
   * @brief Helper function to augment the model by adding frames fixed to the base
   *
   * @param[in] translation Position of the new frame in the base frame
   * @param[in] name Name of the new frame
   * @return pinocchio::FrameIndex Index of the created frame
   */
  pinocchio::FrameIndex addFrameToBase(Eigen::Vector3d translation, std::string name);

  /**
   * @brief Perform a finite difference on the sates.
   *
   * @param[in] x1 Initial state
   * @param[in] x2 Desired state
   * @return Eigen::VectorXd The vector that must be integrated during a unit of time to go from x1 to x2.
   */
  Eigen::VectorXd difference(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);

  /**
   * @brief Compute reduced state from measures by concatenating q,v of the reduced model.
   *
   * @param q Configuration vector of the full model
   * @param v Velocity vector of the full model
   * @return const Eigen::VectorXd State vector of the reduced model.
   */
  Eigen::VectorXd shapeState(const Eigen::VectorXd &q, const Eigen::VectorXd &v);


  // Const getters
  size_t getFootIndex(const std::string &foot_name) const
  {
    std::find(feet_names.begin(), feet_names.end(), foot_name) - feet_names.begin();
  }

  const std::string &getFootName(size_t i) const
  {
    return feet_names.at(i);
  }

  const std::vector<std::string> &getFeetNames() const
  {
    return feet_names;
  }

  FrameIndex getFootId(const std::string &foot_name) const
  {
    return feet_ids.at(getFootIndex(foot_name));
  }

  FrameIndex getRefFootId(const std::string &foot_name) const
  {
    return feet_ids.at(getFootIndex(foot_name));
  }

  double getMass() const
  {
    return mass;
  }

  const Model &getModel()
  {
    return model;
  }

  const Model &getCompleteModel()
  {
    return model_full;
  }
};

class RobotDataHandler {
private:
  RobotModelHandler model_handler;
  Data data;

public:
  RobotDataHandler(const RobotModelHandler &model_handler);

  // Set new robot state
  void updateInternalData(const Eigen::VectorXd &x, const bool updateJacobians);
  void updateJacobiansMassMatrix(const Eigen::VectorXd &x);

  // Const getters
  const SE3 &getRefFootPose(const std::string &foot_name) const
  {
    return data.oMf[model_handler.getRefFootId(foot_name)];
  };
  const SE3 &getFootPose(const std::string &foot_name) const
  {
    return data.oMf[model_handler.getFootId(foot_name)];
  };
  const SE3 &getRootFramePose() const {
    return data.oMf[model_handler.root_id];
  }
  const RobotModelHandler &getModelHandler() const
  {
    return model_handler;
  }
  const Data &getData() const
  {
    return data;
  }
  Eigen::VectorXd getCentroidalState() const;
};

} // namespace simple_mpc
