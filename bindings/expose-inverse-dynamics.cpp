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
      Eigen::VectorXd tau_res(self.model_handler_.getModel().nv - 6);
      self.solve(t, q_meas, v_meas, tau_res);
      return tau_res;
    }

    Eigen::VectorXd getAccelerationsProxy(KinodynamicsID & self)
    {
      Eigen::VectorXd a(self.model_handler_.getModel().nv - 6);
      self.getAccelerations(a);
      return a;
    }

    void exposeInverseDynamics()
    {
      bp::class_<KinodynamicsID::Settings>("KinodynamicsIDSettings", bp::init<>(bp::args("self")))
        .def_readwrite("friction_coefficient", &KinodynamicsID::Settings::friction_coefficient)
        .def_readwrite("contact_weight_ratio_max", &KinodynamicsID::Settings::contact_weight_ratio_max)
        .def_readwrite("contact_weight_ratio_min", &KinodynamicsID::Settings::contact_weight_ratio_min)
        .def_readwrite("kp_base", &KinodynamicsID::Settings::kp_base)
        .def_readwrite("kp_posture", &KinodynamicsID::Settings::kp_posture)
        .def_readwrite("kp_contact", &KinodynamicsID::Settings::kp_contact)
        .def_readwrite("w_base", &KinodynamicsID::Settings::w_base)
        .def_readwrite("w_posture", &KinodynamicsID::Settings::w_posture)
        .def_readwrite("w_contact_motion", &KinodynamicsID::Settings::w_contact_motion)
        .def_readwrite("w_contact_force", &KinodynamicsID::Settings::w_contact_force);

      bp::class_<KinodynamicsID>(
        "KinodynamicsID", bp::init<const simple_mpc::RobotModelHandler &, double, const KinodynamicsID::Settings>(
                            bp::args("self", "model_handler", "control_dt", "settings")))
        .def("setTarget", &KinodynamicsID::setTarget)
        .def("solve", &solveProxy)
        .def("getAccelerations", &getAccelerationsProxy);
    }
  } // namespace python
} // namespace simple_mpc
