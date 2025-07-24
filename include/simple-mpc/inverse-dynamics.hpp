#pragma once

#include <simple-mpc/robot-handler.hpp>
#include <tsid/contacts/contact-point.hpp>
#include <tsid/formulations/inverse-dynamics-formulation-acc-force.hpp>
#include <tsid/solvers/solver-HQP-factory.hxx>
#include <tsid/solvers/utils.hpp>
#include <tsid/tasks/task-joint-bounds.hpp>
#include <tsid/tasks/task-joint-posture.hpp>
#include <tsid/tasks/task-se3-equality.hpp>
#include <tsid/trajectories/trajectory-euclidian.hpp>

namespace simple_mpc
{

  class KinodynamicsID
  {
  public:
    struct Settings
    {
      double kp_posture = 1.0;
      double kp_base = 0.0;
      double kp_contact = 0.0;

      double w_posture = 1e2;
      double w_base = 0.;
      double w_constact_force_ = 0.;
      double w_contact_motion_ = 0.;

      static Settings Default()
      {
        return {};
      } // Work-around c++ bug to have a default constructor
    };

    KinodynamicsID(const simple_mpc::RobotModelHandler & model_handler, const Settings settings = Settings::Default())
    : settings_(settings)
    , model_handler_(model_handler)
    , data_handler_(model_handler_)
    , robot_(model_handler_.getModel())
    , formulation_("tsid", robot_)
    {
      const pinocchio::Model & model = model_handler.getModel();
      const size_t nq = model.nq;
      const size_t nv = model.nv;
      const size_t nu = nv - 6;

      formulation_.computeProblemData(0, model_handler.getReferenceState().head(nq), Eigen::VectorXd::Zero(nv));

      // Prepare contact point task
      // const Eigen::Vector3d normal {0, 0, 1};
      // for (std::string foot_name : model_handler_.getFeetNames()) {
      //   std::shared_ptr<tsid::contacts::ContactPoint> cp =
      //     std::make_shared<tsid::contacts::ContactPoint>(foot_name, robot_, foot_name, normal, 0.3, 1.0, 100.);
      //   cp->Kp(settings_.kp_contact * Eigen::VectorXd::Ones(3));
      //   cp->Kd(2.0 * cp->Kp().cwiseSqrt());
      //   cp->setReference(data_handler_.getFootPose(foot_name));
      //   cp->useLocalFrame(false);
      //   formulation_.addRigidContact(*cp, settings_.w_constact_force_, settings_.w_contact_motion_, 0);
      //   contactPoints_.push_back(cp);
      // }

      // Add the posture task
      postureTask_ = std::make_shared<tsid::tasks::TaskJointPosture>("task-posture", robot_);
      postureTask_->Kp(settings_.kp_posture * Eigen::VectorXd::Ones(nu));
      postureTask_->Kd(2.0 * postureTask_->Kp().cwiseSqrt());
      formulation_.addMotionTask(*postureTask_, settings_.w_posture, 1);

      samplePosture_ = tsid::trajectories::TrajectorySample(robot_.nq_actuated(), robot_.na());

      // Add the base task
      baseTask_ = std::make_shared<tsid::tasks::TaskSE3Equality>("task-base", robot_, "root_joint");
      baseTask_->Kp(settings_.kp_base * Eigen::VectorXd::Ones(6));
      baseTask_->Kd(2.0 * baseTask_->Kp().cwiseSqrt());
      baseTask_->setReference(pose_base_);
      // formulation_.addMotionTask(*baseTask_, settings_.w_base, 0);

      sampleBase_ = tsid::trajectories::TrajectorySample(12, 6);

      // Create an HQP solver
      solver_ = tsid::solvers::SolverHQPFactory::createNewSolver(tsid::solvers::SOLVER_HQP_PROXQP, "solver-proxqp");
      solver_->resize(formulation_.nVar(), formulation_.nEq(), formulation_.nIn());

      HQPData_ = formulation_.computeProblemData(
        0, model_handler.getReferenceState().head(nq), model_handler.getReferenceState().tail(nv));
      sol_ = solver_->solve(HQPData_);
    }

    void setTarget(
      const Eigen::VectorXd & q_target,
      const Eigen::VectorXd & v_target,
      const Eigen::VectorXd & a_target,
      const std::vector<bool> & contact_state,
      const Eigen::VectorXd & f_target)
    {
      samplePosture_.setValue(q_target.tail(robot_.nq_actuated()));
      samplePosture_.setDerivative(v_target.tail(robot_.na()));
      samplePosture_.setSecondDerivative(a_target.tail(robot_.na()));
      postureTask_->setReference(samplePosture_);

      pose_base_.rotation() = pinocchio::SE3::Quaternion(q_target[3], q_target[4], q_target[5], q_target[6]).matrix();
      pose_base_.translation() = q_target.head(3);
      tsid::math::SE3ToVector(pose_base_, sampleBase_.pos);
      sampleBase_.setDerivative(v_target.head(6));
      sampleBase_.setSecondDerivative(a_target.head(6));
      baseTask_->setReference(sampleBase_);

      // for (std::size_t i = 0; i < model_handler_.getFeetNames().size(); i ++) {
      //   std::string name = model_handler_.getFeetNames()[i];
      //   if (contact_state[i]) {
      //     if(!check_contact(name)) {
      //       // formulation_.addRigidContact(*contactPoints_[i], w_constact_force_, w_contact_motion_, 0);
      //     }
      //     else {
      //       contactPoints_[i]->setForceReference(forces.segment(i * 3, 3));
      //       contactPoints_[i]->setReference(robot_->framePosition(data, robot_->model().getFrameId(name)));
      //     }
      //   }
      //   else {
      //     if (check_contact(name)) {
      //       // formulation_.removeRigidContact(name, 0);
      //     }
      //   }
      // }
    }

    bool check_contact(std::string & name)
    {
      for (auto & it : formulation_.m_contacts)
      {
        if (it->contact.name() == name)
        {
          return true;
        }
      }
      return false;
    }

    void
    solve(const double t, const Eigen::VectorXd & q_meas, const Eigen::VectorXd & v_meas, Eigen::VectorXd & tau_res)
    {
      HQPData_ = formulation_.computeProblemData(t, q_meas, v_meas);
      sol_ = solver_->solve(HQPData_);
      tau_res = formulation_.getActuatorForces(sol_);
    }

    // Order matters to be instanciated in the right order
    const Settings settings_;
    const simple_mpc::RobotModelHandler & model_handler_;
    simple_mpc::RobotDataHandler data_handler_;
    tsid::robots::RobotWrapper robot_;
    tsid::InverseDynamicsFormulationAccForce formulation_;

    std::vector<std::shared_ptr<tsid::contacts::ContactPoint>> contactPoints_;
    std::shared_ptr<tsid::tasks::TaskJointPosture> postureTask_;
    std::shared_ptr<tsid::tasks::TaskSE3Equality> baseTask_;
    tsid::solvers::SolverHQPBase * solver_;
    tsid::solvers::HQPData HQPData_;
    tsid::solvers::HQPOutput sol_;
    tsid::trajectories::TrajectorySample samplePosture_; // TODO: no need to store it
    tsid::trajectories::TrajectorySample sampleBase_;    // TODO: no need to store it
    pinocchio::SE3 pose_base_;
  };

} // namespace simple_mpc
