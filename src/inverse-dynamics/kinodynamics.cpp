#include <simple-mpc/inverse-dynamics/kinodynamics.hpp>
#include <tsid/contacts/contact-6d.hpp>
#include <tsid/contacts/contact-point.hpp>

using namespace simple_mpc;

KinodynamicsID::KinodynamicsID(const RobotModelHandler & model_handler, double control_dt, const Settings settings)
: settings_(settings)
, model_handler_(model_handler)
, data_handler_(model_handler_)
, robot_(model_handler_.getModel())
, formulation_("tsid", robot_)
{
  const pinocchio::Model & model = model_handler.getModel();
  const size_t nq = model.nq;
  const size_t nq_actuated = robot_.nq_actuated();
  const size_t nv = model.nv;
  const size_t nu = nv - 6;

  // Prepare foot contact tasks
  const size_t n_contacts = model_handler_.getFeetFrameNames().size();
  const Eigen::Vector3d normal{0, 0, 1};
  const double weight = model_handler_.getMass() * 9.81;
  const double max_f = settings_.contact_weight_ratio_max * weight;
  const double min_f = settings_.contact_weight_ratio_min * weight;
  for (int i = 0; i < n_contacts; i++)
  {
    const std::string frame_name = model_handler.getFootFrameName(i);
    switch (model_handler.getFootType(i))
    {
    case RobotModelHandler::FootType::POINT: {
      auto contact_point = std::make_shared<tsid::contacts::ContactPoint>(
        frame_name, robot_, frame_name, normal, settings_.friction_coefficient, min_f, max_f);
      contact_point->Kp(settings_.kp_contact * Eigen::VectorXd::Ones(3));
      contact_point->Kd(2.0 * contact_point->Kp().cwiseSqrt());
      contact_point->useLocalFrame(false);
      tsid_contacts.push_back(contact_point);
      break;
    }
    case RobotModelHandler::FootType::QUAD: {
      auto contact_6D = std::make_shared<tsid::contacts::Contact6d>(
        frame_name, robot_, frame_name, model_handler_.getQuadFootContactPoints(i).transpose(), normal,
        settings_.friction_coefficient, min_f, max_f);
      contact_6D->Kp(settings_.kp_contact * Eigen::VectorXd::Ones(6));
      contact_6D->Kd(2.0 * contact_6D->Kp().cwiseSqrt());
      tsid_contacts.push_back(contact_6D);
      break;
    }
    default: {
      assert(false);
    }
    }
    // By default contact is not active (will be by setTarget)
    active_tsid_contacts_.push_back(false);
  }

  // Add the posture task
  postureTask_ = std::make_shared<tsid::tasks::TaskJointPosture>("task-posture", robot_);
  postureTask_->Kp(settings_.kp_posture * Eigen::VectorXd::Ones(nu));
  postureTask_->Kd(2.0 * postureTask_->Kp().cwiseSqrt());
  if (settings_.w_posture > 0.)
    formulation_.addMotionTask(*postureTask_, settings_.w_posture, 1);

  samplePosture_ = tsid::trajectories::TrajectorySample(nq_actuated, nu);

  // Add the base task
  baseTask_ = std::make_shared<tsid::tasks::TaskSE3Equality>("task-base", robot_, model_handler_.getBaseFrameName());
  baseTask_->Kp(settings_.kp_base * Eigen::VectorXd::Ones(6));
  baseTask_->Kd(2.0 * baseTask_->Kp().cwiseSqrt());
  if (settings_.w_base > 0.)
    formulation_.addMotionTask(*baseTask_, settings_.w_base, 1);

  sampleBase_ = tsid::trajectories::TrajectorySample(12, 6);

  // Add joint limit task
  boundsTask_ = std::make_shared<tsid::tasks::TaskJointPosVelAccBounds>("task-joint-limits", robot_, control_dt);
  boundsTask_->setPositionBounds(
    model_handler_.getModel().lowerPositionLimit.tail(nq_actuated),
    model_handler_.getModel().upperPositionLimit.tail(nq_actuated));
  boundsTask_->setVelocityBounds(model_handler_.getModel().upperVelocityLimit.tail(nu));
  boundsTask_->setImposeBounds(
    true, true, true, false); // For now do not impose acceleration bound as it is not provided in URDF
  formulation_.addMotionTask(*boundsTask_, 1.0, 0); // No weight needed as it is set as constraint

  // Add actuation limit task
  actuationTask_ = std::make_shared<tsid::tasks::TaskActuationBounds>("actuation-limits", robot_);
  actuationTask_->setBounds(
    model_handler_.getModel().lowerEffortLimit.tail(nu), model_handler_.getModel().upperEffortLimit.tail(nu));
  formulation_.addActuationTask(*actuationTask_, 1.0, 0); // No weight needed as it is set as constraint

  // Create an HQP solver
  solver_ = tsid::solvers::SolverHQPFactory::createNewSolver(tsid::solvers::SOLVER_HQP_PROXQP, "solver-proxqp");
  solver_->resize(formulation_.nVar(), formulation_.nEq(), formulation_.nIn());

  // By default initialize target in reference state
  const Eigen::VectorXd q_ref = model_handler.getReferenceState().head(nq);
  const Eigen::VectorXd v_ref = model_handler.getReferenceState().tail(nv);
  std::vector<bool> c_ref(n_contacts);
  std::vector<TargetContactForce> f_ref;
  for (int i = 0; i < n_contacts; i++)
  {
    // By default initialize all foot in contact with same amount of force
    c_ref[i] = true;
    const RobotModelHandler::FootType foot_type = model_handler.getFootType(i);
    if (foot_type == RobotModelHandler::POINT)
      f_ref.push_back(TargetContactForce::Zero(3));
    else if (foot_type == RobotModelHandler::QUAD)
      f_ref.push_back(TargetContactForce::Zero(6));
    else
      assert(false);
    f_ref[i][2] = weight / n_contacts; // Weight on Z axis
  }
  setTarget(q_ref, v_ref, v_ref, c_ref, f_ref);

  // Dry run to initialize solver data & output
  const tsid::solvers::HQPData & solver_data = formulation_.computeProblemData(0, q_ref, v_ref);
  last_solution_ = solver_->solve(solver_data);
}

void KinodynamicsID::setTarget(
  const Eigen::Ref<const Eigen::VectorXd> & q_target,
  const Eigen::Ref<const Eigen::VectorXd> & v_target,
  const Eigen::Ref<const Eigen::VectorXd> & a_target,
  const std::vector<bool> & contact_state_target,
  const std::vector<TargetContactForce> & f_target)
{
  data_handler_.updateInternalData(q_target, v_target, false);

  // Posture task
  samplePosture_.setValue(q_target.tail(robot_.nq_actuated()));
  samplePosture_.setDerivative(v_target.tail(robot_.na()));
  samplePosture_.setSecondDerivative(a_target.tail(robot_.na()));
  postureTask_->setReference(samplePosture_);

  // Base task
  tsid::math::SE3ToVector(data_handler_.getBaseFramePose(), sampleBase_.pos);
  sampleBase_.setDerivative(v_target.head<6>());
  sampleBase_.setSecondDerivative(a_target.head<6>());
  baseTask_->setReference(sampleBase_);

  // Foot contacts
  for (std::size_t foot_nb = 0; foot_nb < model_handler_.getFeetNb(); foot_nb++)
  {
    const std::string & name{model_handler_.getFootFrameName(foot_nb)};
    if (contact_state_target[foot_nb])
    {
      if (!active_tsid_contacts_[foot_nb])
      {
        formulation_.addRigidContact(
          *tsid_contacts[foot_nb], settings_.w_contact_force, settings_.w_contact_motion,
          settings_.contact_motion_equality ? 0 : 1);
      }
      switch (model_handler_.getFootType(foot_nb))
      {
      case RobotModelHandler::FootType::POINT: {
        std::static_pointer_cast<tsid::contacts::ContactPoint>(tsid_contacts[foot_nb])
          ->setForceReference(f_target.at(foot_nb));
        break;
      }
      case RobotModelHandler::FootType::QUAD: {
        std::static_pointer_cast<tsid::contacts::Contact6d>(tsid_contacts[foot_nb])
          ->setForceReference(f_target.at(foot_nb));
        break;
      }
      default: {
        assert(false);
      }
      }
      tsid_contacts[foot_nb];
      active_tsid_contacts_[foot_nb] = true;
    }
    else
    {
      if (active_tsid_contacts_[foot_nb])
      {
        formulation_.removeRigidContact(name, 0);
        active_tsid_contacts_[foot_nb] = false;
      }
    }
  }
  solver_->resize(formulation_.nVar(), formulation_.nEq(), formulation_.nIn());
}

void KinodynamicsID::solve(
  const double t,
  const Eigen::Ref<const Eigen::VectorXd> & q_meas,
  const Eigen::Ref<const Eigen::VectorXd> & v_meas,
  Eigen::Ref<Eigen::VectorXd> tau_res)
{
  // Update contact position based on the real robot foot placement
  data_handler_.updateInternalData(q_meas, v_meas, false);
  for (std::size_t foot_nb = 0; foot_nb < model_handler_.getFeetNb(); foot_nb++)
  {
    if (active_tsid_contacts_[foot_nb])
    {
      switch (model_handler_.getFootType(foot_nb))
      {
      case RobotModelHandler::FootType::POINT: {
        std::static_pointer_cast<tsid::contacts::ContactPoint>(tsid_contacts[foot_nb])
          ->setReference(data_handler_.getFootPose(foot_nb));
        break;
      }
      case RobotModelHandler::FootType::QUAD: {
        std::static_pointer_cast<tsid::contacts::Contact6d>(tsid_contacts[foot_nb])
          ->setReference(data_handler_.getFootPose(foot_nb));
        break;
      }
      default: {
        assert(false);
      }
      }
    }
  }

  const tsid::solvers::HQPData & solver_data = formulation_.computeProblemData(t, q_meas, v_meas);
  last_solution_ = solver_->solve(solver_data);
  assert(last_solution_.status == tsid::solvers::HQPStatus::HQP_STATUS_OPTIMAL);
  tau_res = formulation_.getActuatorForces(last_solution_);
}

void KinodynamicsID::getAccelerations(Eigen::Ref<Eigen::VectorXd> ddq)
{
  ddq = formulation_.getAccelerations(last_solution_);
}
