///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "simple-mpc/python.hpp"
#include <eigenpy/deprecation-policy.hpp>
#include <eigenpy/std-vector.hpp>

#include "simple-mpc/qp-solvers.hpp"
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/dense/wrapper.hpp>

namespace simple_mpc
{
  namespace python
  {
    namespace bp = boost::python;

    template<typename T>
    inline void py_list_to_std_vector(const bp::object & iterable, std::vector<T> & out)
    {
      out = std::vector<T>(bp::stl_input_iterator<T>(iterable), bp::stl_input_iterator<T>());
    }

    auto * create_idsolver(const bp::dict & settings, const pinocchio::Model & model)
    {
      IDSettings conf;

      py_list_to_std_vector(settings["contact_ids"], conf.contact_ids);

      conf.mu = bp::extract<double>(settings["mu"]);
      conf.Lfoot = bp::extract<double>(settings["Lfoot"]);
      conf.Wfoot = bp::extract<double>(settings["Wfoot"]);

      conf.force_size = bp::extract<int>(settings["force_size"]);
      conf.kd = bp::extract<double>(settings["kd"]);
      conf.w_force = bp::extract<double>(settings["w_force"]);
      conf.w_acc = bp::extract<double>(settings["w_acc"]);
      conf.w_tau = bp::extract<double>(settings["w_tau"]);
      conf.verbose = bp::extract<bool>(settings["verbose"]);

      return new IDSolver(conf, model);
    }

    auto * create_ikidsolver(const bp::dict & settings, const pinocchio::Model & model)
    {
      IKIDSettings conf;

      py_list_to_std_vector(settings["Kp_gains"], conf.Kp_gains);
      py_list_to_std_vector(settings["Kd_gains"], conf.Kd_gains);
      py_list_to_std_vector(settings["contact_ids"], conf.contact_ids);
      py_list_to_std_vector(settings["fixed_frame_ids"], conf.fixed_frame_ids);

      conf.x0 = bp::extract<Eigen::VectorXd>(settings["x0"]);
      conf.dt = bp::extract<double>(settings["dt"]);
      conf.mu = bp::extract<double>(settings["mu"]);
      conf.Lfoot = bp::extract<double>(settings["Lfoot"]);
      conf.Wfoot = bp::extract<double>(settings["Wfoot"]);

      conf.force_size = bp::extract<int>(settings["force_size"]);
      conf.w_qref = bp::extract<double>(settings["w_qref"]);
      conf.w_footpose = bp::extract<double>(settings["w_footpose"]);
      conf.w_centroidal = bp::extract<double>(settings["w_centroidal"]);
      conf.w_baserot = bp::extract<double>(settings["w_baserot"]);
      conf.w_force = bp::extract<double>(settings["w_force"]);
      conf.verbose = bp::extract<bool>(settings["verbose"]);

      return new IKIDSolver(conf, model);
    }

    /// \brief Visitor used to expose common elements in a low-level QP object.
    struct ll_qp_visitor : bp::def_visitor<ll_qp_visitor>
    {
      template<class T, class... PyArgs>
      void visit(bp::class_<T, PyArgs...> & cl) const
      {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        cl.def("getA", &T::getA, "self"_a)
          .def("getH", &T::getH, "self"_a)
          .def("getC", &T::getC, "self"_a)
          .def("getb", &T::getb, "self"_a)
          .def("getg", &T::getg, "self"_a)
          .def_readonly("qp", &T::qp_)
          .def(
            "getQP", &T::getQP, eigenpy::deprecated_member<>("Deprecated getter. Please use the `qp` class attribute."),
            "self"_a)
          .add_property("solved_acc", &T::solved_acc_)
          .add_property("solved_forces", &T::solved_forces_)
          .add_property("solved_torque", &T::solved_torque_);
#pragma GCC diagnostic pop
      };
    };

    void exposeIDSolver()
    {
      eigenpy::StdVectorPythonVisitor<std::vector<pinocchio::SE3>, true>::expose("StdVec_SE3"),
        eigenpy::details::overload_base_get_item_for_std_vector<std::vector<pinocchio::SE3>>();
      bp::class_<IDSolver>("IDSolver", bp::no_init)
        .def(bp::init<const IDSettings &, const pin::Model &>(("self"_a, "settings", "model")))
        .def("__init__", bp::make_constructor(&create_idsolver))
        .def("solveQP", &IDSolver::solveQP, ("self"_a, "data", "contact_state", "v", "a", "tau", "forces", "M"))
        .def(ll_qp_visitor());
    }

    void exposeIKIDSolver()
    {
      eigenpy::StdVectorPythonVisitor<std::vector<pinocchio::SE3>, true>::expose("StdVec_SE3"),
        eigenpy::details::overload_base_get_item_for_std_vector<std::vector<pinocchio::SE3>>();
      bp::class_<IKIDSolver>("IKIDSolver", bp::no_init)
        .def(bp::init<const IKIDSettings &, const pin::Model &>(("self"_a, "settings", "model")))
        .def("__init__", bp::make_constructor(&create_ikidsolver))
        .def(
          "solve_qp", &IKIDSolver::solve_qp,
          bp::args("self", "data", "contact_state", "x_measured", "forces", "dH", "M"))
        .def(
          "computeDifferences", &IKIDSolver::computeDifferences,
          ("self"_a, "data", "x_measured", "foot_refs", "foot_refs_next"))
        .def(ll_qp_visitor());
    }

  } // namespace python
} // namespace simple_mpc
