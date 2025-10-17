///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2025, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/bindings/python/utils/pickle-map.hpp>
#include <pinocchio/fwd.hpp>

#include <eigenpy/deprecation-policy.hpp>
#include <eigenpy/eigenpy.hpp>
#include <eigenpy/std-unique-ptr.hpp>
#include <eigenpy/std-vector.hpp>

#include "simple-mpc/arm-dynamics.hpp"
#include "simple-mpc/arm-mpc.hpp"
#include "simple-mpc/python.hpp"

namespace simple_mpc
{
  namespace python
  {
    namespace bp = boost::python;
    using eigenpy::StdVectorPythonVisitor;

    ArmMPC * createArmMPC(const bp::dict & settings, std::shared_ptr<ArmDynamicsOCP> problem)
    {
      ArmMPCSettings conf;

      conf.TOL = bp::extract<double>(settings["TOL"]);
      conf.mu_init = bp::extract<double>(settings["mu_init"]);
      conf.max_iters = bp::extract<std::size_t>(settings["max_iters"]);
      conf.num_threads = bp::extract<std::size_t>(settings["num_threads"]);
      conf.timestep = bp::extract<double>(settings["timestep"]);

      return new ArmMPC{conf, problem};
    }

    bp::dict getSettings(ArmMPC & self)
    {
      ArmMPCSettings & conf = self.settings_;
      bp::dict settings;
      settings["TOL"] = conf.TOL;
      settings["mu_init"] = conf.mu_init;
      settings["max_iters"] = conf.max_iters;
      settings["num_threads"] = conf.num_threads;
      settings["timestep"] = conf.timestep;

      return settings;
    }

    void exposeArmMPC()
    {
      using StageVec = std::vector<std::shared_ptr<StageModel>>;
      StdVectorPythonVisitor<StageVec, true>::expose(
        "StdVec_StageModel", eigenpy::details::overload_base_get_item_for_std_vector<StageVec>());

      bp::class_<ArmMPC, boost::noncopyable>("ArmMPC", bp::no_init)
        .def("__init__", bp::make_constructor(&createArmMPC, bp::default_call_policies()))
        .def("getSettings", &getSettings)
        .def("iterate", &ArmMPC::iterate, bp::args("self", "x"))
        .def("setReferencePose", &ArmMPC::setReferencePose, bp::args("self", "t", "pose_ref"))
        .def("getReferencePose", &ArmMPC::getReferencePose, bp::args("self", "t"))
        .def_readwrite("x_reference", &ArmMPC::x_reference_)
        .def_readonly("ocp_handler", &ArmMPC::ocp_handler_)
        .def("switchToReach", &ArmMPC::switchToReach, ("self"_a, "reach_pose"))
        .def("switchToRest", &ArmMPC::switchToRest, "self"_a)
        .def("getStateDerivative", &ArmMPC::getStateDerivative, ("self"_a, "t"))
        .def(
          "getModelHandler", &ArmMPC::getModelHandler, "self"_a, bp::return_internal_reference<>(),
          "Get the robot model handler.")
        .def(
          "getDataHandler", &ArmMPC::getDataHandler, "self"_a, bp::return_internal_reference<>(),
          "Get the robot data handler.")
        .def(
          "getTrajOptProblem", &ArmMPC::getTrajOptProblem, "self"_a, bp::return_internal_reference<>(),
          "Get the trajectory optimal problem.")
        .add_property("solver", bp::make_getter(&ArmMPC::solver_, eigenpy::ReturnInternalStdUniquePtr{}))
        .add_property("xs", &ArmMPC::xs_)
        .add_property("us", &ArmMPC::us_)
        .add_property("Ks", &ArmMPC::Ks_);
    }

  } // namespace python
} // namespace simple_mpc
