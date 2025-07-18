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

    Eigen::VectorXd interpolateConfigurationProxy(
      Interpolator & self, const double delay, const double timestep, const std::vector<Eigen::VectorXd> & qs)
    {
      Eigen::VectorXd q_interp(qs[0].size());
      self.interpolateConfiguration(delay, timestep, qs, q_interp);

      return q_interp;
    }

    Eigen::VectorXd interpolateStateProxy(
      Interpolator & self, const double delay, const double timestep, const std::vector<Eigen::VectorXd> & xs)
    {
      Eigen::VectorXd x_interp(xs[0].size());
      self.interpolateState(delay, timestep, xs, x_interp);

      return x_interp;
    }

    Eigen::VectorXd interpolateLinearProxy(
      Interpolator & self, const double delay, const double timestep, const std::vector<Eigen::VectorXd> & vs)
    {
      Eigen::VectorXd v_interp(vs[0].size());
      self.interpolateLinear(delay, timestep, vs, v_interp);

      return v_interp;
    }

    std::vector<bool> interpolateContactsProxy(
      Interpolator & self, const double delay, const double timestep, const std::vector<std::vector<bool>> & cs)
    {
      std::vector<bool> c_interp(cs[0].size());
      self.interpolateContacts(delay, timestep, cs, c_interp);

      return c_interp;
    }

    void exposeInterpolator()
    {
      bp::class_<Interpolator>("Interpolator", bp::init<const Model &>(bp::args("self", "model")))
        .def("interpolateConfiguration", &interpolateConfigurationProxy)
        .def("interpolateState", &interpolateStateProxy)
        .def("interpolateLinear", &interpolateLinearProxy)
        .def("interpolateContacts", &interpolateContactsProxy);
    }
  } // namespace python
} // namespace simple_mpc
