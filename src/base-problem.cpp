///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "simple-mpc/base-problem.hpp"

namespace simple_mpc {
using namespace aligator;

Problem::Problem(const RobotHandler &handler) : handler_(handler) {
  nq_ = handler_.get_rmodel().nq;
  nv_ = handler_.get_rmodel().nv;
  nu_ = nv_ - 6;
}

void Problem::create_problem(const Eigen::VectorXd &x0,
                             const std::vector<ContactMap> &contact_sequence) {
  std::vector<xyz::polymorphic<StageModel>> stage_models;
  for (auto cm : contact_sequence) {
    std::vector<bool> contact_states = cm.getContactStates();
    std::vector<Eigen::VectorXd> force_ref;
    for (std::size_t i = 0; i < contact_states.size(); i++) {
      force_ref.push_back(Eigen::VectorXd::Zero(6));
    }
    stage_models.push_back(create_stage(cm, force_ref));
  }

  problem_ = std::make_shared<TrajOptProblem>(x0, stage_models,
                                              create_terminal_cost());
}

void Problem::insert_cost(CostStack &cost_stack,
                          const xyz::polymorphic<CostAbstract> &cost,
                          std::map<std::string, std::size_t> &cost_map,
                          const std::string &name, int &cost_incr) {
  cost_stack.addCost(cost);
  cost_map.insert({"name", cost_incr});
  cost_incr++;
}

} // namespace simple_mpc
