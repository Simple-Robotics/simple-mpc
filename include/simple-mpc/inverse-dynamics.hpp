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
#define DEFINE_FIELD(type, name, value)                                                                                \
  type name = value;                                                                                                   \
  Settings & set_##name(type v)                                                                                        \
  {                                                                                                                    \
    name = v;                                                                                                          \
    return *this;                                                                                                      \
  }

      // Physical quantities
      DEFINE_FIELD(double, friction_coefficient, 0.3)
      DEFINE_FIELD(
        double,
        contact_weight_ratio_max,
        2.0) // Max force for one foot contact (express as a multiple of the robot weight)
      DEFINE_FIELD(
        double,
        contact_weight_ratio_min,
        0.01) // Min force for one foot contact (express as a multiple of the robot weight)

      // Tasks gains
      DEFINE_FIELD(double, kp_posture, 1.0)
      DEFINE_FIELD(double, kp_base, 0.0)
      DEFINE_FIELD(double, kp_contact, 0.0)

      // Tasks weights
      DEFINE_FIELD(double, w_posture, 1e2)
      DEFINE_FIELD(double, w_base, 0.)
      DEFINE_FIELD(double, w_contact_force, 0.)
      DEFINE_FIELD(double, w_contact_motion, 0.)

      static Settings Default()
      {
        return {};
      } // Work-around c++ bug to have a default constructor of nested class
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
      const size_t n_contacts = model_handler_.getFeetNames().size();
      const Eigen::Vector3d normal{0, 0, 1};
      const double weight = model_handler_.getMass() * 9.81;
      const double max_f = settings_.contact_weight_ratio_max * weight;
      const double min_f = settings_.contact_weight_ratio_min * weight;
      for (int i = 0; i < n_contacts; i++)
      {
        const std::string frame_name = model_handler.getFootName(i);
        // Create contact point
        tsid::contacts::ContactPoint & contact_point = tsid_contacts.emplace_back(
          frame_name, robot_, frame_name, normal, settings_.friction_coefficient, min_f, max_f);
        // Set contact parameters
        contact_point.Kp(settings_.kp_contact * Eigen::VectorXd::Ones(3));
        contact_point.Kd(2.0 * contact_point.Kp().cwiseSqrt());
        contact_point.useLocalFrame(false);
        // By default contact is not active (will be by setTarget)
        active_tsid_contacts_.push_back(false);
      }

      // Add the posture task
      postureTask_ = std::make_shared<tsid::tasks::TaskJointPosture>("task-posture", robot_);
      postureTask_->Kp(settings_.kp_posture * Eigen::VectorXd::Ones(nu));
      postureTask_->Kd(2.0 * postureTask_->Kp().cwiseSqrt());
      formulation_.addMotionTask(*postureTask_, settings_.w_posture, 1);

      samplePosture_ = tsid::trajectories::TrajectorySample(robot_.nq_actuated(), robot_.na());

      // Add the base task
      baseTask_ =
        std::make_shared<tsid::tasks::TaskSE3Equality>("task-base", robot_, model_handler_.getBaseFrameName());
      baseTask_->Kp(settings_.kp_base * Eigen::VectorXd::Ones(6));
      baseTask_->Kd(2.0 * baseTask_->Kp().cwiseSqrt());
      baseTask_->setReference(pose_base_);
      formulation_.addMotionTask(*baseTask_, settings_.w_base, 1);

      sampleBase_ = tsid::trajectories::TrajectorySample(12, 6);

      // Create an HQP solver
      solver_ = tsid::solvers::SolverHQPFactory::createNewSolver(tsid::solvers::SOLVER_HQP_PROXQP, "solver-proxqp");
      solver_->resize(formulation_.nVar(), formulation_.nEq(), formulation_.nIn());

      // Dry run to initialize solver data & output
      const tsid::solvers::HQPData & solver_data = formulation_.computeProblemData(
        0, model_handler.getReferenceState().head(nq), model_handler.getReferenceState().tail(nv));
      last_solution_ = solver_->solve(solver_data);
    }

    void setTarget(
      const Eigen::VectorXd & q_target,
      const Eigen::VectorXd & v_target,
      const Eigen::VectorXd & a_target,
      const std::vector<bool> & contact_state_target,
      const Eigen::VectorXd & f_target)
    {
      // Posture task
      samplePosture_.setValue(q_target.tail(robot_.nq_actuated()));
      samplePosture_.setDerivative(v_target.tail(robot_.na()));
      samplePosture_.setSecondDerivative(a_target.tail(robot_.na()));
      postureTask_->setReference(samplePosture_);

      // Base task
      pose_base_.rotation() = pinocchio::SE3::Quaternion(q_target[3], q_target[4], q_target[5], q_target[6]).matrix();
      pose_base_.translation() = q_target.head(3);
      tsid::math::SE3ToVector(pose_base_, sampleBase_.pos);
      sampleBase_.setDerivative(v_target.head(6));
      sampleBase_.setSecondDerivative(a_target.head(6));
      baseTask_->setReference(sampleBase_);

      // Foot contacts
      data_handler_.updateInternalData(q_target, v_target, false);
      for (std::size_t i = 0; i < model_handler_.getFeetNames().size(); i++)
      {
        std::string name = model_handler_.getFeetNames()[i];
        if (contact_state_target[i])
        {
          if (!active_tsid_contacts_[i])
          {
            formulation_.addRigidContact(tsid_contacts[i], settings_.w_contact_force, settings_.w_contact_motion, 1);
          }
          tsid_contacts[i].setForceReference(f_target.segment(i * 3, 3));
          tsid_contacts[i].setReference(data_handler_.getFootPose(i));
          active_tsid_contacts_[i] = true;
        }
        else
        {
          if (active_tsid_contacts_[i])
          {
            formulation_.removeRigidContact(name, 0);
            active_tsid_contacts_[i] = false;
          }
        }
      }
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
      const tsid::solvers::HQPData & solver_data_ = formulation_.computeProblemData(t, q_meas, v_meas);
      last_solution_ = solver_->solve(solver_data_);
      tau_res = formulation_.getActuatorForces(last_solution_);
    }

    const Eigen::VectorXd & getAccelerations()
    {
      return formulation_.getAccelerations(last_solution_);
    }

  private:
    // Order matters to be instanciated in the right order
    const Settings settings_;
    const simple_mpc::RobotModelHandler & model_handler_;
    simple_mpc::RobotDataHandler data_handler_;
    tsid::robots::RobotWrapper robot_;
    tsid::InverseDynamicsFormulationAccForce formulation_;

    std::vector<bool> active_tsid_contacts_;
    std::vector<tsid::contacts::ContactPoint> tsid_contacts;
    std::shared_ptr<tsid::tasks::TaskJointPosture> postureTask_;
    std::shared_ptr<tsid::tasks::TaskSE3Equality> baseTask_;
    tsid::solvers::SolverHQPBase * solver_;
    tsid::solvers::HQPOutput last_solution_;
    tsid::trajectories::TrajectorySample samplePosture_; // TODO: no need to store it
    tsid::trajectories::TrajectorySample sampleBase_;    // TODO: no need to store it
    pinocchio::SE3 pose_base_;
  };

} // namespace simple_mpc
