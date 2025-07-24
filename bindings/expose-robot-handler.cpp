///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <eigenpy/eigenpy.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/multibody/fwd.hpp>

#include "simple-mpc/robot-handler.hpp"

namespace simple_mpc
{
  namespace python
  {
    namespace bp = boost::python;

    void exposeHandler()
    {
      bp::class_<RobotModelHandler>(
        "RobotModelHandler", bp::init<const pinocchio::Model &, const std::string &, const std::string &>(
                               bp::args("self", "model", "reference_configuration_name", "base_frame_name")))
        .def("addFoot", &RobotModelHandler::addFoot)
        .def("setFootReferencePlacement", &RobotModelHandler::setFootReferencePlacement)
        .def("difference", &RobotModelHandler::difference)
        .def("getBaseFrameId", &RobotModelHandler::getBaseFrameId)
        .def("getReferenceState", &RobotModelHandler::getReferenceState)
        .def("getFootNb", &RobotModelHandler::getFootNb)
        .def("getFeetIds", &RobotModelHandler::getFeetIds, bp::return_internal_reference<>())
        .def("getFootName", &RobotModelHandler::getFootName, bp::return_internal_reference<>())
        .def("getFeetNames", &RobotModelHandler::getFeetNames, bp::return_internal_reference<>())
        .def("getFootId", &RobotModelHandler::getFootId)
        .def("getRefFootId", &RobotModelHandler::getRefFootId)
        .def("getMass", &RobotModelHandler::getMass)
        .def("getModel", &RobotModelHandler::getModel, bp::return_internal_reference<>());

      ENABLE_SPECIFIC_MATRIX_TYPE(RobotDataHandler::CentroidalStateVector);

      bp::class_<RobotDataHandler>(
        "RobotDataHandler", bp::init<const RobotModelHandler &>(bp::args("self", "model_handler")))
        // .def("updateInternalData", &RobotDataHandler::updateInternalData)
        .def("updateJacobiansMassMatrix", &RobotDataHandler::updateJacobiansMassMatrix)
        // .def("getRefFootPose", &RobotDataHandler::getRefFootPose, bp::return_internal_reference<>())
        // .def("getFootPose", &RobotDataHandler::getFootPose, bp::return_internal_reference<>())
        .def("getBaseFramePose", &RobotDataHandler::getBaseFramePose, bp::return_internal_reference<>())
        .def("getModelHandler", &RobotDataHandler::getModelHandler, bp::return_internal_reference<>())
        .def("getData", &RobotDataHandler::getData, bp::return_internal_reference<>())
        .def("getCentroidalState", &RobotDataHandler::getCentroidalState)
        .def("getState", &RobotDataHandler::getState);
      return;
    }

  } // namespace python
} // namespace simple_mpc
