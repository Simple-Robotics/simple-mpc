#include "simple-mpc/ocp-handler.hpp"

namespace simple_mpc
{
  using namespace aligator;

  OCPHandler::~OCPHandler()
  {
  }

  OCPHandler::OCPHandler(const RobotModelHandler & model_handler)
  : model_handler_(model_handler)
  , problem_(nullptr)
  {
    nq_ = model_handler.getModel().nq;
    nv_ = model_handler.getModel().nv;
    ndx_ = 2 * model_handler.getModel().nv;
    nu_ = nv_ - 6;
  }

  std::vector<xyz::polymorphic<StageModel>> OCPHandler::createStages(
    const std::vector<std::map<std::string, bool>> & contact_phases,
    const std::vector<std::map<std::string, pinocchio::SE3>> & contact_poses,
    const std::vector<std::map<std::string, Eigen::VectorXd>> & contact_forces)
  {
    if (contact_phases.size() != contact_poses.size())
    {
      throw std::runtime_error("Contact phases and poses sequences do not have the same size");
    }
    if (contact_phases.size() != contact_forces.size())
    {
      throw std::runtime_error("Contact phases and forces sequences do not have the same size");
    }
    std::map<std::string, bool> previous_phases;
    for (auto const & name : model_handler_.getFeetNames())
    {
      previous_phases.insert({name, true});
    }
    std::vector<xyz::polymorphic<StageModel>> stage_models;
    for (std::size_t i = 0; i < contact_phases.size(); i++)
    {
      std::map<std::string, bool> land_constraint;
      for (auto const & name : model_handler_.getFeetNames())
      {
        if (!previous_phases.at(name) and contact_phases[i].at(name))
          land_constraint.insert({name, true});
        else
          land_constraint.insert({name, false});
      }
      StageModel stage = createStage(contact_phases[i], contact_poses[i], contact_forces[i], land_constraint);
      stage_models.push_back(std::move(stage));
      previous_phases = contact_phases[i];
    }

    return stage_models;
  }

  void OCPHandler::setReferenceControl(const std::size_t t, const ConstVectorRef & u_ref)
  {
    CostStack * cs = getCostStack(t);
    QuadraticControlCost * qc = cs->getComponent<QuadraticControlCost>("control_cost");
    qc->setTarget(u_ref);
  }

  ConstVectorRef OCPHandler::getReferenceControl(const std::size_t t)
  {
    CostStack * cs = getCostStack(t);
    QuadraticControlCost * qc = cs->getComponent<QuadraticControlCost>("control_cost");
    return qc->getTarget();
  }

  CostStack * OCPHandler::getCostStack(std::size_t t)
  {
    if (t >= getSize())
    {
      throw std::runtime_error("Stage index exceeds stage vector size");
    }
    CostStack * cs = dynamic_cast<CostStack *>(&*problem_->stages_[t]->cost_);

    return cs;
  }

  CostStack * OCPHandler::getTerminalCostStack()
  {
    CostStack * cs = dynamic_cast<CostStack *>(&*problem_->term_cost_);

    return cs;
  }

  std::size_t OCPHandler::getCostNumber() const
  {
    CostStack * cs = dynamic_cast<CostStack *>(&*problem_->stages_[0]->cost_);
    return cs->components_.size();
  }

  void OCPHandler::createProblem(
    const ConstVectorRef & x0,
    const size_t horizon,
    const int force_size,
    const double gravity,
    const bool terminal_constraint = false)
  {
    std::vector<std::map<std::string, bool>> contact_phases;
    std::vector<std::map<std::string, pinocchio::SE3>> contact_poses;
    std::vector<std::map<std::string, Eigen::VectorXd>> contact_forces;

    Eigen::VectorXd force_ref(force_size);
    force_ref.setZero();
    force_ref[2] = -model_handler_.getMass() * gravity / (double)model_handler_.getFeetNames().size();

    std::map<std::string, bool> contact_phase;
    std::map<std::string, pinocchio::SE3> contact_pose;
    std::map<std::string, Eigen::VectorXd> contact_force;
    for (auto & name : model_handler_.getFeetNames())
    {
      contact_phase.insert({name, true});
      contact_pose.insert({name, pinocchio::SE3::Identity()});
      contact_force.insert({name, force_ref});
    }

    for (size_t i = 0; i < horizon; i++)
    {
      contact_phases.push_back(contact_phase);
      contact_poses.push_back(contact_pose);
      contact_forces.push_back(contact_force);
    }
    std::vector<xyz::polymorphic<StageModel>> stage_models =
      createStages(contact_phases, contact_poses, contact_forces);

    problem_ = std::make_unique<TrajOptProblem>(x0, std::move(stage_models), createTerminalCost());
    problem_initialized_ = true;

    if (terminal_constraint)
    {
      createTerminalConstraint(x0.head(3));
    }
  }
} // namespace simple_mpc
