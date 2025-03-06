///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <aligator/modelling/centroidal/angular-acceleration.hpp>
#include <aligator/modelling/centroidal/angular-momentum.hpp>
#include <aligator/modelling/centroidal/centroidal-acceleration.hpp>
#include <aligator/modelling/centroidal/centroidal-translation.hxx>
#include <aligator/modelling/centroidal/linear-momentum.hpp>
#include <aligator/modelling/dynamics/centroidal-fwd.hpp>

#include "simple-mpc/fwd.hpp"
#include "simple-mpc/ocp-handler.hpp"

namespace simple_mpc
{
  using namespace aligator;

  /**
   * @brief Build a centroidal dynamics problem based on
   * the CentroidalFwdDynamics from Aligator.
   *
   * State is defined as concatenation of center of mass position and
   * centroidal momentum; control is defined as concatenation of
   * contact forces.
   */

  struct CentroidalSettings
  {
    // timestep in problem shooting nodes
    double timestep;

    // Cost function weights
    Eigen::MatrixXd w_u;           // Control
    Eigen::Matrix3d w_com;         // Linear momentum
    Eigen::Matrix3d w_linear_mom;  // Linear momentum
    Eigen::Matrix3d w_angular_mom; // Angular momentum
    Eigen::Matrix3d w_linear_acc;  // Linear acceleration
    Eigen::Matrix3d w_angular_acc; // Angular acceleration

    // Physics parameters
    Eigen::Vector3d gravity;
    double mu;
    double Lfoot;
    double Wfoot;
    int force_size;
  };

  class CentroidalOCP : public OCPHandler
  {
    using Base = OCPHandler;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    explicit CentroidalOCP(const CentroidalSettings & settings, const RobotModelHandler & model_handler);

    SIMPLE_MPC_DEFINE_DEFAULT_MOVE_CTORS(CentroidalOCP);

    virtual ~CentroidalOCP() {};

    // Create one Centroidal stage
    StageModel createStage(
      const std::map<std::string, bool> & contact_phase,
      const std::map<std::string, pinocchio::SE3> & contact_pose,
      const std::map<std::string, Eigen::VectorXd> & contact_force,
      const std::map<std::string, bool> & land_constraint) override;

    // Manage terminal cost and constraint
    CostStack createTerminalCost() override;
    void createTerminalConstraint(const Eigen::Vector3d & com_ref) override;
    void updateTerminalConstraint(const Eigen::Vector3d & com_ref) override;

    // Getters and setters for pose not implemented
    void setReferencePose(const std::size_t t, const std::string & ee_name, const pinocchio::SE3 & pose_ref) override;
    void setReferencePoses(const std::size_t t, const std::map<std::string, pinocchio::SE3> & pose_refs) override;
    void setTerminalReferencePose(const std::string & /*ee_name*/, const pinocchio::SE3 & /*pose_ref*/) override
    {
    }
    const pinocchio::SE3 getReferencePose(const std::size_t t, const std::string & ee_name) override;

    // Getters and setters for contact forces
    void setReferenceForces(const std::size_t t, const std::map<std::string, Eigen::VectorXd> & force_refs) override;
    void setReferenceForce(const std::size_t t, const std::string & ee_name, const ConstVectorRef & force_ref) override;
    const Eigen::VectorXd getReferenceForce(const std::size_t t, const std::string & ee_name) override;
    void computeControlFromForces(const std::map<std::string, Eigen::VectorXd> & force_refs);
    const Eigen::VectorXd getVelocityBase(const std::size_t t) override;
    void setVelocityBase(const std::size_t t, const ConstVectorRef & velocity_base) override;
    const Eigen::VectorXd getPoseBase(const std::size_t t) override;
    void setPoseBase(const std::size_t t, const ConstVectorRef & pose_base) override;
    const Eigen::VectorXd getProblemState(const RobotDataHandler & data_handler) override;
    size_t getContactSupport(const std::size_t t) override;
    std::vector<bool> getContactState(const std::size_t t) override;
    void setReferenceState(const std::size_t t, const ConstVectorRef & x_ref) override;
    const ConstVectorRef getReferenceState(const std::size_t t) override;

    CentroidalSettings getSettings()
    {
      return settings_;
    }

  protected:
    CentroidalSettings settings_;
    int nx_;
    Eigen::Vector3d com_ref_;
  };

} // namespace simple_mpc
