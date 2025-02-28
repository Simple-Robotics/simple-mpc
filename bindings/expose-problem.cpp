///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <aligator/modelling/contact-map.hpp>
#include <eigenpy/eigenpy.hpp>
#include <eigenpy/std-map.hpp>
#include <eigenpy/std-vector.hpp>

#include "simple-mpc/python/py-ocp-handler.hpp"

#include "simple-mpc/fwd.hpp"

namespace simple_mpc
{
  namespace python
  {
    namespace bp = boost::python;
    using eigenpy::StdVectorPythonVisitor;
    using eigenpy::python::StdMapPythonVisitor;

    void exposeContainers()
    {
      StdMapPythonVisitor<
        std::string, Eigen::VectorXd, std::less<std::string>,
        std::allocator<std::pair<const std::string, Eigen::VectorXd>>, true>::expose("StdMap_Force");

      StdMapPythonVisitor<
        std::string, pinocchio::SE3, std::less<std::string>,
        std::allocator<std::pair<const std::string, pinocchio::SE3>>, true>::expose("StdMap_SE3");

      StdMapPythonVisitor<
        std::string, bool, std::less<std::string>, std::allocator<std::pair<const std::string, bool>>,
        true>::expose("StdMap_bool");

      StdVectorPythonVisitor<std::vector<ContactMap>, true>::expose("StdVec_ContactMap_double");
    }

    void exposeOcpHandler()
    {
      bp::register_ptr_to_python<std::shared_ptr<OCPHandler>>();
      bp::class_<PyOCPHandler, boost::noncopyable>("OCPHandler", bp::no_init)
        .def(bp::init<const RobotModelHandler &>(("self"_a, "model_handler")))
        .def(
          "createStage", bp::pure_virtual(&OCPHandler::createStage),
          ("self"_a, "contact_map", "force_refs", "land_constraint"))
        .def("createTerminalCost", bp::pure_virtual(&OCPHandler::createTerminalCost), "self"_a)
        .def("createTerminalConstraint", bp::pure_virtual(&OCPHandler::createTerminalConstraint), "self"_a)
        .def("updateTerminalConstraint", bp::pure_virtual(&OCPHandler::updateTerminalConstraint), ("self"_a, "com_ref"))
        .def(
          "setReferencePose", bp::pure_virtual(&OCPHandler::setReferencePose),
          bp::args("self", "t", "ee_name", "pose_ref"))
        .def("setReferencePoses", bp::pure_virtual(&OCPHandler::setReferencePoses), bp::args("self", "t", "pose_refs"))
        .def(
          "setTerminalReferencePose", bp::pure_virtual(&OCPHandler::setTerminalReferencePose),
          bp::args("self", "ee_name", "pose_ref"))
        .def("getReferencePose", bp::pure_virtual(&OCPHandler::getReferencePose), bp::args("self", "t", "ee_name"))
        .def(
          "setReferenceForces", bp::pure_virtual(&OCPHandler::setReferenceForces), bp::args("self", "t", "force_refs"))
        .def(
          "setReferenceForce", bp::pure_virtual(&OCPHandler::setReferenceForce),
          bp::args("self", "t", "ee_name", "force_ref"))
        .def("getReferenceForce", bp::pure_virtual(&OCPHandler::getReferenceForce), bp::args("self", "t", "ee_name"))
        .def("setVelocityBase", bp::pure_virtual(&OCPHandler::setVelocityBase), bp::args("self", "t", "velocity_base"))
        .def("getVelocityBase", bp::pure_virtual(&OCPHandler::getVelocityBase), bp::args("self", "t"))
        .def("setPoseBase", bp::pure_virtual(&OCPHandler::setPoseBase), bp::args("self", "t", "pose_base"))
        .def("getPoseBase", bp::pure_virtual(&OCPHandler::getPoseBase), bp::args("self", "t"))
        .def("getProblemState", bp::pure_virtual(&OCPHandler::getProblemState), bp::args("self", "data_handler"))
        .def("getContactSupport", bp::pure_virtual(&OCPHandler::getContactSupport), bp::args("self", "t"))
        .def("getContactState", bp::pure_virtual(&OCPHandler::getContactState), bp::args("self", "t"))
        .def(
          "createProblem", &OCPHandler::createProblem,
          ("self"_a, "x0", "horizon", "force_size", "gravity", "terminal_constraint"))
        .def("setReferenceControl", &OCPHandler::setReferenceControl, ("self"_a, "t", "u_ref"))
        .def("getReferenceControl", &OCPHandler::getReferenceControl, ("self"_a, "t"))
        .def("setReferenceState", &OCPHandler::setReferenceState, ("self"_a, "x_ref"))
        .def("getReferenceState", &OCPHandler::getReferenceState, ("self"_a))
        .def("getProblem", +[](OCPHandler & ocp) { return boost::ref(ocp.getProblem()); }, "self"_a);

      exposeContainers();
    }

  } // namespace python
} // namespace simple_mpc
