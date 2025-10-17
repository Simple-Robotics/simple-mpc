///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2025, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "simple-mpc/arm-dynamics.hpp"
#include "simple-mpc/python.hpp"

#include "simple-mpc/fwd.hpp"
#include <eigenpy/eigenpy.hpp>
#include <eigenpy/std-map.hpp>

namespace simple_mpc::python
{

  auto * createArmDynamics(const bp::dict & settings, const RobotModelHandler & model_handler)
  {
    ArmDynamicsSettings conf;
    conf.timestep = bp::extract<double>(settings["timestep"]);
    conf.w_x = bp::extract<Eigen::MatrixXd>(settings["w_x"]);
    conf.w_u = bp::extract<Eigen::MatrixXd>(settings["w_u"]);
    conf.w_frame = bp::extract<Eigen::MatrixXd>(settings["w_frame"]);

    conf.gravity = bp::extract<Eigen::Vector3d>(settings["gravity"]);

    conf.qmin = bp::extract<Eigen::VectorXd>(settings["qmin"]);
    conf.qmax = bp::extract<Eigen::VectorXd>(settings["qmax"]);

    conf.umin = bp::extract<Eigen::VectorXd>(settings["umin"]);
    conf.umax = bp::extract<Eigen::VectorXd>(settings["umax"]);

    conf.kinematics_limits = bp::extract<bool>(settings["kinematics_limits"]);
    conf.torque_limits = bp::extract<bool>(settings["torque_limits"]);

    conf.ee_name = bp::extract<std::string>(settings["ee_name"]);

    return new ArmDynamicsOCP(conf, model_handler);
  }

  bp::dict getSettingsArm(ArmDynamicsOCP & self)
  {
    ArmDynamicsSettings conf = self.getSettings();
    bp::dict settings;
    settings["timestep"] = conf.timestep;
    settings["w_x"] = conf.w_x;
    settings["w_u"] = conf.w_u;
    settings["w_frame"] = conf.w_frame;
    settings["gravity"] = conf.gravity;
    settings["qmin"] = conf.qmin;
    settings["qmax"] = conf.qmax;
    settings["umin"] = conf.umin;
    settings["umax"] = conf.umax;
    settings["kinematics_limits"] = conf.kinematics_limits;
    settings["torque_limits"] = conf.torque_limits;
    settings["ee_name"] = conf.ee_name;

    return settings;
  }

  void exposeArmDynamicsOcp()
  {
    bp::register_ptr_to_python<shared_ptr<ArmDynamicsOCP>>();

    bp::class_<ArmDynamicsOCP, boost::noncopyable>("ArmDynamicsOCP", bp::no_init)
      .def(
        "__init__",
        bp::make_constructor(&createArmDynamics, bp::default_call_policies(), ("settings"_a, "model_handler")))
      .def("getSettings", &getSettingsArm)
      .def("createStage", &ArmDynamicsOCP::createStage, bp::args("self", "reaching", "reach_pose"))
      .def("createProblem", &ArmDynamicsOCP::createProblem, ("self"_a, "x0", "horizon"))
      .def("setReferencePose", &ArmDynamicsOCP::setReferencePose, bp::args("self", "t", "pose_ref"))
      .def("getReferencePose", &ArmDynamicsOCP::getReferencePose, bp::args("self", "t"))
      .def("setReferenceState", &ArmDynamicsOCP::setReferenceState, bp::args("self", "t", "x_ref"))
      .def("getReferenceState", &ArmDynamicsOCP::getReferenceState, bp::args("self", "t"))
      .def("getProblem", +[](ArmDynamicsOCP & ocp) { return boost::ref(ocp.getProblem()); }, "self"_a);
  }

} // namespace simple_mpc::python
