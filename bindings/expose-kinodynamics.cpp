#include "simple-mpc/kinodynamics.hpp"
#include "simple-mpc/python.hpp"

#include <eigenpy/std-map.hpp>

namespace simple_mpc::python
{

  auto * createKinodynamics(const bp::dict & settings, const RobotModelHandler & model_handler)
  {
    KinodynamicsSettings conf;
    conf.timestep = bp::extract<double>(settings["timestep"]);
    conf.w_x = bp::extract<Eigen::MatrixXd>(settings["w_x"]);
    conf.w_u = bp::extract<Eigen::MatrixXd>(settings["w_u"]);
    conf.w_cent = bp::extract<Eigen::MatrixXd>(settings["w_cent"]);
    conf.w_centder = bp::extract<Eigen::MatrixXd>(settings["w_centder"]);
    conf.w_frame = bp::extract<Eigen::MatrixXd>(settings["w_frame"]);

    conf.gravity = bp::extract<Eigen::Vector3d>(settings["gravity"]);
    conf.force_size = bp::extract<int>(settings["force_size"]);

    conf.qmin = bp::extract<Eigen::VectorXd>(settings["qmin"]);
    conf.qmax = bp::extract<Eigen::VectorXd>(settings["qmax"]);

    conf.mu = bp::extract<double>(settings["mu"]);
    conf.Lfoot = bp::extract<double>(settings["Lfoot"]);
    conf.Wfoot = bp::extract<double>(settings["Wfoot"]);

    conf.kinematics_limits = bp::extract<bool>(settings["kinematics_limits"]);
    conf.force_cone = bp::extract<bool>(settings["force_cone"]);
    conf.land_cstr = bp::extract<bool>(settings["land_cstr"]);

    return new KinodynamicsOCP(conf, model_handler);
  }

  bp::dict getSettingsKino(KinodynamicsOCP & self)
  {
    KinodynamicsSettings conf = self.getSettings();
    bp::dict settings;
    settings["timestep"] = conf.timestep;
    settings["w_x"] = conf.w_x;
    settings["w_u"] = conf.w_u;
    settings["w_cent"] = conf.w_cent;
    settings["w_centder"] = conf.w_centder;
    settings["w_frame"] = conf.w_frame;
    settings["gravity"] = conf.gravity;
    settings["force_size"] = conf.force_size;
    settings["qmin"] = conf.qmin;
    settings["qmax"] = conf.qmax;
    settings["mu"] = conf.mu;
    settings["Lfoot"] = conf.Lfoot;
    settings["Wfoot"] = conf.Wfoot;
    settings["kinematics_limits"] = conf.kinematics_limits;
    settings["force_cone"] = conf.force_cone;
    settings["land_cstr"] = conf.land_cstr;

    return settings;
  }

  StageModel createKinoStage(
    KinodynamicsOCP & self,
    const bp::dict & phase_dict,
    const bp::dict & pose_dict,
    const bp::dict & force_dict,
    const bp::dict & land_dict)
  {
    bp::list phase_keys = bp::list(phase_dict.keys());
    bp::list pose_keys = bp::list(pose_dict.keys());
    bp::list force_keys = bp::list(force_dict.keys());
    bp::list land_keys = bp::list(land_dict.keys());
    std::map<std::string, bool> phase_contact;
    std::map<std::string, pinocchio::SE3> pose_contact;
    std::map<std::string, Eigen::VectorXd> force_contact;
    std::map<std::string, bool> land_constraint;
    for (int i = 0; i < len(phase_keys); ++i)
    {
      bp::extract<std::string> extractor(phase_keys[i]);
      if (extractor.check())
      {
        std::string key = extractor();
        bool ff = bp::extract<bool>(phase_dict[key]);
        phase_contact.insert({key, ff});
      }
    }
    for (int i = 0; i < len(pose_keys); ++i)
    {
      bp::extract<std::string> extractor(pose_keys[i]);
      if (extractor.check())
      {
        std::string key = extractor();
        pinocchio::SE3 ff = bp::extract<pinocchio::SE3>(pose_dict[key]);
        pose_contact.insert({key, ff});
      }
    }
    for (int i = 0; i < len(force_keys); ++i)
    {
      bp::extract<std::string> extractor(force_keys[i]);
      if (extractor.check())
      {
        std::string key = extractor();
        Eigen::VectorXd ff = bp::extract<Eigen::VectorXd>(force_dict[key]);
        force_contact.insert({key, ff});
      }
    }
    for (int i = 0; i < len(land_keys); ++i)
    {
      bp::extract<std::string> extractor(land_keys[i]);
      if (extractor.check())
      {
        std::string key = extractor();
        bool ff = bp::extract<bool>(land_dict[key]);
        land_constraint.insert({key, ff});
      }
    }

    return self.createStage(phase_contact, pose_contact, force_contact, land_constraint);
  }

  void exposeKinodynamicsOcp()
  {
    bp::register_ptr_to_python<shared_ptr<KinodynamicsOCP>>();

    bp::class_<KinodynamicsOCP, bp::bases<OCPHandler>, boost::noncopyable>("KinodynamicsOCP", bp::no_init)
      .def(
        "__init__",
        bp::make_constructor(&createKinodynamics, bp::default_call_policies(), ("settings"_a, "model_handler")))
      .def("getSettings", &getSettingsKino)
      .def("createStage", &createKinoStage);
  }

} // namespace simple_mpc::python
