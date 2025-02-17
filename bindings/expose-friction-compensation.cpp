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

    Eigen::VectorXd
    computeFrictionProxy(FrictionCompensation & self, Eigen::Ref<const VectorXd> velocity, Eigen::Ref<VectorXd> torque)
    {
      self.computeFriction(velocity, torque);

      return torque;
    }

    void exposeFrictionCompensation()
    {
      bp::class_<FrictionCompensation>(
        "FrictionCompensation", bp::init<const Model &, const bool>(bp::args("self", "model", "with_free_flyer")))
        .def("computeFriction", &computeFrictionProxy)
        .add_property("dry_friction", &FrictionCompensation::dry_friction_)
        .add_property("viscuous_friction", &FrictionCompensation::viscuous_friction_);
    }

  } // namespace python
} // namespace simple_mpc
