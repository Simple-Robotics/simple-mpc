#include <eigenpy/eigenpy.hpp>

#include "simple-mpc/inverse-dynamics/centroidal.hpp"
#include "simple-mpc/inverse-dynamics/kinodynamics.hpp"

namespace simple_mpc
{
  namespace python
  {
    namespace bp = boost::python;

    Eigen::VectorXd solveKinoProxy(
      KinodynamicsID & self,
      const double t,
      const Eigen::Ref<const Eigen::VectorXd> & q_meas,
      const Eigen::Ref<const Eigen::VectorXd> & v_meas)
    {
      Eigen::VectorXd tau_res(self.model_handler_.getModel().nv - 6);
      self.solve(t, q_meas, v_meas, tau_res);
      return tau_res;
    }

    Eigen::VectorXd getAccelerationsKinoProxy(KinodynamicsID & self)
    {
      Eigen::VectorXd a(self.model_handler_.getModel().nv);
      self.getAccelerations(a);
      return a;
    }

    Eigen::VectorXd solveCentroidalProxy(
      CentroidalID & self,
      const double t,
      const Eigen::Ref<const Eigen::VectorXd> & q_meas,
      const Eigen::Ref<const Eigen::VectorXd> & v_meas)
    {
      Eigen::VectorXd tau_res(self.model_handler_.getModel().nv - 6);
      self.solve(t, q_meas, v_meas, tau_res);
      return tau_res;
    }

    Eigen::VectorXd getAccelerationsCentroidalProxy(CentroidalID & self)
    {
      Eigen::VectorXd a(self.model_handler_.getModel().nv);
      self.getAccelerations(a);
      return a;
    }

    void setTarget_CentroidalID(
      CentroidalID & self,
      const Eigen::Ref<const Eigen::Vector<double, 3>> & com_position,
      const Eigen::Ref<const Eigen::Vector<double, 3>> & com_velocity,
      const CentroidalID::FeetPoseVector & feet_pose,
      const CentroidalID::FeetVelocityVector & feet_velocity,
      const std::vector<bool> & contact_state_target,
      const std::vector<CentroidalID::TargetContactForce> & f_target)
    {
      self.setTarget(com_position, com_velocity, feet_pose, feet_velocity, contact_state_target, f_target);
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
        .def_readwrite("w_contact_force", &KinodynamicsID::Settings::w_contact_force)
        .def_readwrite("contact_motion_equality", &KinodynamicsID::Settings::contact_motion_equality);

      bp::class_<KinodynamicsID>(
        "KinodynamicsID", bp::init<const simple_mpc::RobotModelHandler &, double, const KinodynamicsID::Settings>(
                            bp::args("self", "model_handler", "control_dt", "settings")))
        .def("setTarget", &KinodynamicsID::setTarget)
        .def("solve", &solveKinoProxy)
        .def("getAccelerations", &getAccelerationsKinoProxy);

      bp::class_<CentroidalID::Settings, bp::bases<KinodynamicsID::Settings>>(
        "CentroidalIDSettings", bp::init<>(bp::args("self")))
        .def_readwrite("kp_com", &CentroidalID::Settings::kp_com)
        .def_readwrite("kp_feet_tracking", &CentroidalID::Settings::kp_feet_tracking)
        .def_readwrite("w_com", &CentroidalID::Settings::w_com)
        .def_readwrite("w_feet_tracking", &CentroidalID::Settings::w_feet_tracking);

      bp::class_<CentroidalID>(
        "CentroidalID", bp::init<const simple_mpc::RobotModelHandler &, double, const CentroidalID::Settings>(
                          bp::args("self", "model_handler", "control_dt", "settings")))
        .def("setTarget", &setTarget_CentroidalID)
        .def("solve", &solveCentroidalProxy)
        .def("getAccelerations", &getAccelerationsCentroidalProxy);
    }
  } // namespace python
} // namespace simple_mpc
