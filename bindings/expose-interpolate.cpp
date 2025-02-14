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

    Eigen::VectorXd stateInterpolateProxy(
      StateInterpolator & self, const double delay, const double timestep, const std::vector<Eigen::VectorXd> xs)
    {
      Eigen::VectorXd x_interp(xs[0].size());
      self.interpolate(delay, timestep, xs, x_interp);

      return x_interp;
    }

    Eigen::VectorXd linearInterpolateProxy(
      LinearInterpolator & self, const double delay, const double timestep, const std::vector<Eigen::VectorXd> xs)
    {
      Eigen::VectorXd x_interp(xs[0].size());
      self.interpolate(delay, timestep, xs, x_interp);

      return x_interp;
    }

    void exposeStateInterpolator()
    {
      bp::class_<StateInterpolator>("StateInterpolator", bp::init<const Model &>(bp::args("self", "model")))
        .def("interpolate", &stateInterpolateProxy);
    }

    void exposeLinearInterpolator()
    {
      bp::class_<LinearInterpolator>("LinearInterpolator", bp::init<const size_t>(bp::args("self", "vec_size")))
        .def("interpolate", &linearInterpolateProxy);
    }

  } // namespace python
} // namespace simple_mpc
