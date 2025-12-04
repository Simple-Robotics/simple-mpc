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
      ENABLE_SPECIFIC_MATRIX_TYPE(RobotModelHandler::ContactPointsMatrix);

      bp::class_<RobotModelHandler>(
        "RobotModelHandler", bp::init<const pinocchio::Model &, const std::string &, const std::string &>(
                               bp::args("self", "model", "reference_configuration_name", "base_frame_name")))
        .def("addPointFoot", &RobotModelHandler::addPointFoot)
        .def("addQuadFoot", &RobotModelHandler::addQuadFoot)
        .def("setFootReferencePlacement", &RobotModelHandler::setFootReferencePlacement)
        .def("difference", &RobotModelHandler::difference)
        .def("getBaseFrameName", &RobotModelHandler::getBaseFrameName)
        .def("getBaseFrameId", &RobotModelHandler::getBaseFrameId)
        .def("getReferenceState", &RobotModelHandler::getReferenceState)
        .def("getFeetNb", &RobotModelHandler::getFeetNb)
        .def("getFootNb", &RobotModelHandler::getFootNb)
        .def("getFeetFrameIds", &RobotModelHandler::getFeetFrameIds, bp::return_internal_reference<>())
        .def("getFootFrameName", &RobotModelHandler::getFootFrameName, bp::return_internal_reference<>())
        .def("getFeetFrameNames", &RobotModelHandler::getFeetFrameNames, bp::return_internal_reference<>())
        .def("getFootFrameId", &RobotModelHandler::getFootFrameId)
        .def("getFootRefFrameId", &RobotModelHandler::getFootRefFrameId)
        .def("getMass", &RobotModelHandler::getMass)
        .def("getModel", &RobotModelHandler::getModel, bp::return_internal_reference<>());

      ENABLE_SPECIFIC_MATRIX_TYPE(RobotDataHandler::CentroidalStateVector);

      bp::class_<RobotDataHandler>(
        "RobotDataHandler", bp::init<const RobotModelHandler &>(bp::args("self", "model_handler")))
        .def(
          "updateInternalData", static_cast<void (RobotDataHandler::*)(const ConstVectorRef &, const bool)>(
                                  &RobotDataHandler::updateInternalData))
        .def("updateJacobiansMassMatrix", &RobotDataHandler::updateJacobiansMassMatrix)
        .def("getFootRefPose", &RobotDataHandler::getFootRefPose, bp::return_internal_reference<>())
        .def("getFootPose", &RobotDataHandler::getFootPose, bp::return_internal_reference<>())
        .def("getBaseFramePose", &RobotDataHandler::getBaseFramePose, bp::return_internal_reference<>())
        .def("getModelHandler", &RobotDataHandler::getModelHandler, bp::return_internal_reference<>())
        .def("getData", &RobotDataHandler::getData, bp::return_internal_reference<>())
        .def("getCentroidalState", &RobotDataHandler::getCentroidalState)
        .def("getState", &RobotDataHandler::getState);
      return;
    }

  } // namespace python
} // namespace simple_mpc
