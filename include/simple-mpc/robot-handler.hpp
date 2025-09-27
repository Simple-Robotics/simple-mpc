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
#include <string>
#include <vector>

#include "simple-mpc/fwd.hpp"

namespace simple_mpc
{
  using namespace pinocchio;

  /**
   * @brief Class managing every robot-related quantities.
   *
   * It holds the robot data, controlled joints, end-effector names
   * and other useful items.
   */
  struct RobotModelHandler
  {
    enum FootType
    {
      POINT,
      QUAD
    };
    typedef Eigen::Matrix<double, 4, 3> ContactPointsMatrix;

  public:
    /**
     * @brief Construct a new Robot Model Handler object
     *
     * @param model Model of the full robot
     * @param feet_names Name of the frames corresponding to the feet (e.g. can be
     * used for contact with the ground)
     * @param reference_configuration_name Reference configuration to use
     */
    RobotModelHandler(
      const Model & model, const std::string & reference_configuration_name, const std::string & base_frame_name);

    /**
     * @brief Create a point foot, that can apply 3D force to the ground. (in comparison to 6D foot)
     *
     * @param foot_name Frame name that will be used a a foot
     * @param reference_parent_frame_name Frame to which the foot reference
     * frame will be attached.
     *
     * @return the foot number
     *
     * @note The foot placement will be set by default using the reference configuration of the RobotModelHandler
     */
    size_t addPointFoot(const std::string & foot_name, const std::string & reference_parent_frame_name);

    /**
     * @brief Create a point foot, that can apply 6D wrench to the ground. (in comparison to point foot)
     *
     * @param foot_name Frame name that will be used a a foot
     * @param reference_parent_frame_name Frame to which the foot reference
     * frame will be attached.
     * @param contactPoints 3D positions (in the local frame) of the foot 4 extremum point
     *
     * @return the foot number
     *
     * @note The foot placement will be set by default using the reference configuration of the RobotModelHandler
     */
    size_t addQuadFoot(
      const std::string & foot_name,
      const std::string & reference_parent_frame_name,
      const ContactPointsMatrix & contactPoints);

    /**
     * @brief Update the placement of the foot reference frame wrt to the joint/frame it is attached to
     *
     * @param foot_name Foot frame name
     * @param parentframeMfootref Placement of the foot reference frame wrt the parent frame it is attached to.
     */
    void setFootReferencePlacement(size_t foot_nb, const SE3 & parentframeMfootref);

    /**
     * @brief Perform a finite difference on the sates.
     *
     * @param[in] x1 Initial state
     * @param[in] x2 Desired state
     * @return Eigen::VectorXd The vector that must be integrated during a unit of
     * time to go from x1 to x2.
     */
    Eigen::VectorXd difference(const ConstVectorRef & x1, const ConstVectorRef & x2) const;

    // Const getters
    ConstVectorRef getReferenceState() const
    {
      return reference_state_;
    }

    const std::string & getFootFrameName(size_t foot_nb) const
    {
      return feet_frame_names_.at(foot_nb);
    }

    size_t getFeetNb() const
    {
      return size(feet_frame_ids_);
    }

    size_t getFootNb(const std::string & foot_frame_name) const
    {
      return size_t(
        std::find(feet_frame_names_.begin(), feet_frame_names_.end(), foot_frame_name) - feet_frame_names_.begin());
    }

    FootType getFootType(size_t foot_nb) const
    {
      return feet_types_[foot_nb];
    }

    const ContactPointsMatrix & getQuadFootContactPoints(size_t foot_nb) const
    {
      return feet_contact_points_.at(foot_nb);
    }

    const std::vector<FrameIndex> & getFeetFrameIds() const
    {
      return feet_frame_ids_;
    }

    const std::vector<std::string> & getFeetFrameNames() const
    {
      return feet_frame_names_;
    }

    FrameIndex getBaseFrameId() const
    {
      return base_frame_id_;
    }

    std::string getBaseFrameName() const
    {
      return model_.frames[base_frame_id_].name;
    }

    FrameIndex getFootFrameId(size_t foot_nb) const
    {
      return feet_frame_ids_.at(foot_nb);
    }

    FrameIndex getFootRefFrameId(size_t foot_nb) const
    {
      return feet_ref_frame_ids_.at(foot_nb);
    }

    double getMass() const
    {
      return mass_;
    }

    const Model & getModel() const
    {
      return model_;
    }

  private:
    /**
     * @brief Common operation to perform to add a foot (of any type) : Create the reference frame and set it default
     * pose using the default pose of the robot
     *
     * @param foot_name Name of the frame foot
     * @param reference_parent_frame_name Name of the frame where the reference frame of the foot will be attached
     */
    void addFootFrames(const std::string & foot_name, const std::string & reference_parent_frame_name);

  private:
    /**
     * @brief Model to be used by ocp
     */
    Model model_;

    /**
     * @brief Robot total mass
     */
    double mass_;

    /**
     * @brief Reference configuration and velocity (most probably null velocity)
     * to use
     */
    Eigen::VectorXd reference_state_;

    /**
     * @brief Names of the frames to be in contact with the environment
     */
    std::vector<std::string> feet_frame_names_;

    /**
     * @brief Is the foot a contact point or a 6D contact
     */
    std::vector<FootType> feet_types_;

    /**
     * @brief List of contact points for each 6D feets
     */
    std::map<size_t, Eigen::Matrix<double, 4, 3>> feet_contact_points_;

    /**
     * @brief Ids of the frames to be in contact with the environment
     */
    std::vector<FrameIndex> feet_frame_ids_;

    /**
     * @brief Ids of the frames that are reference position for the feet
     */
    std::vector<FrameIndex> feet_ref_frame_ids_;

    /**
     * @brief Base frame id
     */
    pinocchio::FrameIndex base_frame_id_;
  };

  class RobotDataHandler
  {
  public:
    typedef Eigen::Matrix<double, 9, 1> CentroidalStateVector;

  private:
    RobotModelHandler model_handler_;
    Data data_;
    Eigen::VectorXd x_;

  public:
    RobotDataHandler(const RobotModelHandler & model_handler);

    // Set new robot state
    void updateInternalData(const ConstVectorRef & q, const ConstVectorRef & v, const bool updateJacobians);
    void updateInternalData(const ConstVectorRef & x, const bool updateJacobians);
    void updateJacobiansMassMatrix(const ConstVectorRef & x);

    // Const getters
    const SE3 & getFootRefPose(size_t foot_nb) const
    {
      return data_.oMf[model_handler_.getFootRefFrameId(foot_nb)];
    };
    const SE3 & getFootPose(size_t foot_nb) const
    {
      return data_.oMf[model_handler_.getFootFrameId(foot_nb)];
    };
    const SE3 & getBaseFramePose() const
    {
      return data_.oMf[model_handler_.getBaseFrameId()];
    }
    const RobotModelHandler & getModelHandler() const
    {
      return model_handler_;
    }
    const Data & getData() const
    {
      return data_;
    }
    const Eigen::VectorXd getState() const
    {
      return x_;
    };
    RobotDataHandler::CentroidalStateVector getCentroidalState() const;
  };

} // namespace simple_mpc
