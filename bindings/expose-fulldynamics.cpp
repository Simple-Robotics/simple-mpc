#include "simple-mpc/fulldynamics.hpp"
#include "simple-mpc/python/py-ocp-handler.hpp"

#include <eigenpy/std-map.hpp>
#include <eigenpy/std-vector.hpp>

namespace simple_mpc::python {

void initializeFull(FullDynamicsProblem &self, const bp::dict &settings) {
  FullDynamicsSettings conf;
  conf.timestep = bp::extract<double>(settings["timestep"]);
  conf.w_x = bp::extract<Eigen::MatrixXd>(settings["w_x"]);
  conf.w_u = bp::extract<Eigen::MatrixXd>(settings["w_u"]);
  conf.w_cent = bp::extract<Eigen::MatrixXd>(settings["w_cent"]);
  conf.w_forces = bp::extract<Eigen::MatrixXd>(settings["w_forces"]);
  conf.w_frame = bp::extract<Eigen::MatrixXd>(settings["w_frame"]);

  conf.gravity = bp::extract<Eigen::Vector3d>(settings["gravity"]);
  conf.force_size = bp::extract<int>(settings["force_size"]);
  /// Foot parameters
  conf.mu = bp::extract<double>(settings["mu"]);
  conf.Lfoot = bp::extract<double>(settings["Lfoot"]);
  conf.Wfoot = bp::extract<double>(settings["Wfoot"]);

  /// Limits
  conf.umin = bp::extract<Eigen::VectorXd>(settings["umin"]);
  conf.umax = bp::extract<Eigen::VectorXd>(settings["umax"]);

  conf.qmin = bp::extract<Eigen::VectorXd>(settings["qmin"]);
  conf.qmax = bp::extract<Eigen::VectorXd>(settings["qmax"]);

  /// Baumgarte correctors
  conf.Kp_correction = bp::extract<Eigen::VectorXd>(settings["Kp_correction"]);
  conf.Kd_correction = bp::extract<Eigen::VectorXd>(settings["Kd_correction"]);

  /// Constraints
  conf.torque_limits = bp::extract<bool>(settings["torque_limits"]);
  conf.kinematics_limits = bp::extract<bool>(settings["kinematics_limits"]);
  conf.force_cone = bp::extract<bool>(settings["force_cone"]);

  self.initialize(conf);
}

StageModel createFullStage(FullDynamicsProblem &self,
                           const bp::dict &phase_dict,
                           const bp::dict &pose_dict,
                           const bp::dict &force_dict,
                           const bp::dict &land_dict) {
  bp::list phase_keys(phase_dict.keys());
  bp::list pose_keys(pose_dict.keys());
  bp::list force_keys(force_dict.keys());
  bp::list land_keys(land_dict.keys());
  std::map<std::string, bool> phase_contact;
  std::map<std::string, pinocchio::SE3> pose_contact;
  std::map<std::string, Eigen::VectorXd> force_contact;
  std::map<std::string, bool> land_constraint;
  for (int i = 0; i < len(phase_keys); ++i) {
    bp::extract<std::string> extractor(phase_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      bool ff = bp::extract<bool>(phase_dict[key]);
      phase_contact.insert({key, ff});
    }
  }
  for (int i = 0; i < len(pose_keys); ++i) {
    bp::extract<std::string> extractor(pose_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      pinocchio::SE3 ff = bp::extract<pinocchio::SE3>(pose_dict[key]);
      pose_contact.insert({key, ff});
    }
  }
  for (int i = 0; i < len(force_keys); ++i) {
    bp::extract<std::string> extractor(force_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      Eigen::VectorXd ff = bp::extract<Eigen::VectorXd>(force_dict[key]);
      force_contact.insert({key, ff});
    }
  }
  for (int i = 0; i < len(land_keys); ++i) {
    bp::extract<std::string> extractor(land_keys[i]);
    if (extractor.check()) {
      std::string key = extractor();
      bool ff = bp::extract<bool>(land_dict[key]);
      land_constraint.insert({key, ff});
    }
  }

  return self.createStage(phase_contact, pose_contact, force_contact,
                          land_constraint);
}

bp::dict getSettingsFull(FullDynamicsProblem &self) {
  FullDynamicsSettings conf = self.getSettings();
  bp::dict settings;
  settings["timestep"] = conf.timestep;
  settings["w_x"] = conf.w_x;
  settings["w_u"] = conf.w_u;
  settings["w_cent"] = conf.w_cent;
  settings["gravity"] = conf.gravity;
  settings["force_size"] = conf.force_size;
  settings["w_forces"] = conf.w_forces;
  settings["w_frame"] = conf.w_frame;
  settings["umin"] = conf.umin;
  settings["umax"] = conf.umax;
  settings["qmin"] = conf.qmin;
  settings["qmax"] = conf.qmax;
  settings["Kp_correction"] = conf.Kp_correction;
  settings["Kd_correction"] = conf.Kd_correction;
  settings["mu"] = conf.mu;
  settings["Lfoot"] = conf.Lfoot;
  settings["Wfoot"] = conf.Wfoot;
  settings["torque_limits"] = conf.torque_limits;
  settings["kinematics_limits"] = conf.kinematics_limits;
  settings["force_cone"] = conf.force_cone;

  return settings;
}

void createFullProblem(FullDynamicsProblem &self, const Eigen::VectorXd &x0,
                       const size_t horizon, const int force_size,
                       const double gravity, const bool terminal_constraint) {

  self.createProblem(x0, horizon, force_size, gravity, terminal_constraint);
}

void exposeFullDynamicsProblem() {
  bp::register_ptr_to_python<std::shared_ptr<FullDynamicsProblem>>();

  bp::class_<FullDynamicsProblem, bp::bases<Problem>, boost::noncopyable>(
      "FullDynamicsProblem",
      bp::init<const RobotHandler &>(bp::args("self", "handler")))
      .def("initialize", &initializeFull, bp::args("self", "settings"))
      .def("getSettings", &getSettingsFull)
      .def("initialize",
           bp::make_function(
               &FullDynamicsProblem::initialize,
               bp::return_value_policy<bp::reference_existing_object>()))
      .def("createStage", &createFullStage)
      .def("createProblem", &createFullProblem);
}

} // namespace simple_mpc::python
