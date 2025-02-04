///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2025, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <eigenpy/deprecation-policy.hpp>
#include <eigenpy/std-vector.hpp>

#include "simple-mpc/interpolator.hpp"

namespace simple_mpc
{
  namespace python
  {
    namespace bp = boost::python;

    void exposeInterpolator()
    {
      bp::class_<Interpolator>(
        "Interpolator", bp::init<const long, const long, const long, const long, const double>(
                          bp::args("self", "nx", "nv", "nu", "nu", "MPC_timestep")))
        .def("interpolate", &Interpolator::interpolate)
        .add_property("MPC_timestep", &Interpolator::MPC_timestep_)
        .add_property("x_interpolated", &Interpolator::x_interpolated_)
        .add_property("u_interpolated", &Interpolator::u_interpolated_)
        .add_property("a_interpolated", &Interpolator::a_interpolated_)
        .add_property("forces_interpolated", &Interpolator::forces_interpolated_);
    }

  } // namespace python
} // namespace simple_mpc
