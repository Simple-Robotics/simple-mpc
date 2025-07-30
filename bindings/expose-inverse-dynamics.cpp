#include <eigenpy/eigenpy.hpp>

#include "simple-mpc/inverse-dynamics.hpp"

namespace simple_mpc
{
  namespace python
  {
    namespace bp = boost::python;

    Eigen::VectorXd solveProxy(
      KinodynamicsID & self,
      const double t,
      const Eigen::Ref<const Eigen::VectorXd> & q_meas,
      const Eigen::Ref<const Eigen::VectorXd> & v_meas)
    {
      Eigen::VectorXd tau_res(v_meas.size());
      self.solve(t, q_meas, v_meas, tau_res);
      return tau_res;
    }

    void exposeInverseDynamics()
    {
      bp::class_<KinodynamicsID::Settings>("KinodynamicsIDSettings", bp::init<>(bp::args("self")));

      bp::class_<KinodynamicsID>(
        "KinodynamicsID", bp::init<const simple_mpc::RobotModelHandler &, double, const KinodynamicsID::Settings>(
                            bp::args("self", "model_handler", "control_dt", "settings")))
        .def("setTarget", &KinodynamicsID::setTarget)
        .def("solve", &solveProxy);
    }
  } // namespace python
} // namespace simple_mpc
