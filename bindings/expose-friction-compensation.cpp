///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2025, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <eigenpy/eigenpy.hpp>
#include <pinocchio/multibody/fwd.hpp>

#include "simple-mpc/friction-compensation.hpp"

namespace simple_mpc
{
  namespace python
  {
    namespace bp = boost::python;

    void exposeFrictionCompensation()
    {
      bp::class_<FrictionCompensation>(
        "FrictionCompensation", bp::init<const Model &, const long>(bp::args("self", "model", "actuation_size")))
        .def("computeFriction", &FrictionCompensation::computeFriction)
        .add_property("corrected_torque", &FrictionCompensation::corrected_torque_)
        .add_property("dry_friction", &FrictionCompensation::dry_friction_)
        .add_property("viscuous_friction", &FrictionCompensation::viscuous_friction_);
    }

  } // namespace python
} // namespace simple_mpc
