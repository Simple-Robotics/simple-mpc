///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
/**
 * @file qp-solvers.hpp
 * @brief QP inverse dynamics for MPC schemes
 */

#ifndef SIMPLE_MPC_QP_SOLVERS_HPP_
#define SIMPLE_MPC_QP_SOLVERS_HPP_

#include "simple-mpc/deprecated.hpp"
#include "simple-mpc/fwd.hpp"
#include <pinocchio/multibody/fwd.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>

namespace simple_mpc
{
  using namespace proxsuite;

  struct IDSettings
  {
    std::vector<FrameIndex> contact_ids; //< Index of contacts
    double mu;                           //< Friction parameter
    double Lfoot;                        //< Half-length of foot (if contact 6D)
    double Wfoot;                        //< Half-width of foot (if contact 6D)
    long force_size;                     //< Dimension of contact forces
    double kd;                           //< Baumgarte coefficient
    double w_acc;                        //< Weight for acceleration regularization
    double w_tau;                        //< Weight for torque regularization
    double w_force;                      //< Weight for force regularization
    bool verbose;                        //< Print solver information
  };

  struct IKIDSettings
  {
    std::vector<Eigen::VectorXd> Kp_gains;   //< Proportional gains
    std::vector<Eigen::VectorXd> Kd_gains;   //< Derivative gains
    std::vector<FrameIndex> contact_ids;     //< Index of contacts
    std::vector<FrameIndex> fixed_frame_ids; //< Index of frames kept fixed
    Eigen::VectorXd x0;                      //< Reference state
    double dt;                               //< Integration timestep
    double mu;                               //< Friction parameter
    double Lfoot;                            //< Half-length of foot (if contact 6D)
    double Wfoot;                            //< Half-width of foot (if contact 6D)
    long force_size;                         //< Dimension of contact forces
    double w_qref;                           //< Weight for configuration regularization
    double w_footpose;                       //< Weight for foot placement
    double w_centroidal;                     //< Weight for CoM tracking
    double w_baserot;                        //< Weight for base rotation
    double w_force;                          //< Weight for force regularization
    bool verbose;                            //< Print solver information
  };

  class IDSolver
  {
  public:
    proxqp::dense::QP<double> qp_;

  protected:
    IDSettings settings_;
    pin::Model model_;
    int force_dim_;
    int nforcein_;
    int nk_;

    Eigen::MatrixXd H_;
    Eigen::MatrixXd A_;
    Eigen::MatrixXd C_;
    Eigen::MatrixXd S_;
    Eigen::MatrixXd Cmin_;
    Eigen::VectorXd b_;
    Eigen::VectorXd g_;
    Eigen::VectorXd l_;
    Eigen::VectorXd u_;

    Eigen::MatrixXd Jc_;
    Eigen::VectorXd gamma_;
    Eigen::MatrixXd Jdot_;

    // Internal matrix computation
    void computeMatrices(
      pin::Data & data,
      const std::vector<bool> & contact_state,
      const ConstVectorRef & v,
      const ConstVectorRef & a,
      const ConstVectorRef & tau,
      const ConstVectorRef & forces,
      const ConstMatrixRef & M);

  public:
    explicit IDSolver(const IDSettings & settings, const pin::Model & model);

    void solveQP(
      pin::Data & data,
      const std::vector<bool> & contact_state,
      const ConstVectorRef & v,
      const ConstVectorRef & a,
      const ConstVectorRef & tau,
      const ConstVectorRef & forces,
      const ConstMatrixRef & M);

    SIMPLE_MPC_DEPRECATED
    proxqp::dense::Model<double> getQP()
    {
      return qp_.model;
    }

    Eigen::MatrixXd getA()
    {
      return qp_.model.A;
    }
    Eigen::MatrixXd getH()
    {
      return qp_.model.H;
    }
    Eigen::MatrixXd getC()
    {
      return qp_.model.C;
    }
    Eigen::VectorXd getg()
    {
      return qp_.model.g;
    }
    Eigen::VectorXd getb()
    {
      return qp_.model.b;
    }

    // QP results
    Eigen::VectorXd solved_forces_;
    Eigen::VectorXd solved_acc_;
    Eigen::VectorXd solved_torque_;
  };

  class IKIDSolver
  {
  public:
    proxqp::dense::QP<double> qp_;

  protected:
    IKIDSettings settings_;
    pin::Model model_;
    int force_dim_;
    int nforcein_;
    int nk_;
    int fs_;

    Eigen::MatrixXd H_;
    Eigen::MatrixXd A_;
    Eigen::MatrixXd C_;
    Eigen::MatrixXd S_;
    Eigen::MatrixXd Cmin_;
    Eigen::VectorXd b_;
    Eigen::VectorXd g_;
    Eigen::VectorXd l_;
    Eigen::VectorXd u_;
    Eigen::VectorXd l_box_;
    Eigen::VectorXd u_box_;
    pin::Motion Jvel_;

    Eigen::MatrixXd Jfoot_;
    Eigen::MatrixXd dJfoot_;
    Eigen::MatrixXd Jframe_;
    Eigen::MatrixXd dJframe_;

    std::vector<Eigen::VectorXd> foot_diffs_;
    std::vector<Eigen::VectorXd> dfoot_diffs_;
    std::vector<Eigen::Vector3d> frame_diffs_;
    std::vector<Eigen::Vector3d> dframe_diffs_;
    Eigen::VectorXd q_diff_;
    Eigen::VectorXd dq_diff_;

    // Internal matrix computation
    void computeMatrices(
      pin::Data & data,
      const std::vector<bool> & contact_state,
      const ConstVectorRef & x_measured,
      const ConstVectorRef & forces,
      const ConstVectorRef & dH,
      const ConstMatrixRef & M);

  public:
    explicit IKIDSolver(const IKIDSettings & settings, const pin::Model & model);

    void computeDifferences(
      pin::Data & data,
      const ConstVectorRef & x_measured,
      const std::vector<pin::SE3> & foot_refs,
      const std::vector<pin::SE3> & foot_refs_next);

    void solve_qp(
      pin::Data & data,
      const std::vector<bool> & contact_state,
      const ConstVectorRef & x_measured,
      const ConstVectorRef & forces,
      const ConstVectorRef & dH,
      const ConstMatrixRef & M);

    SIMPLE_MPC_DEPRECATED
    proxqp::dense::Model<double> getQP()
    {
      return qp_.model;
    }

    Eigen::MatrixXd getA()
    {
      return qp_.model.A;
    }
    Eigen::MatrixXd getH()
    {
      return qp_.model.H;
    }
    Eigen::MatrixXd getC()
    {
      return qp_.model.C;
    }
    Eigen::VectorXd getg()
    {
      return qp_.model.g;
    }
    Eigen::VectorXd getb()
    {
      return qp_.model.b;
    }

    // QP results
    Eigen::VectorXd solved_forces_;
    Eigen::VectorXd solved_acc_;
    Eigen::VectorXd solved_torque_;
  };

} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_QP_SOLVERS_HPP_
