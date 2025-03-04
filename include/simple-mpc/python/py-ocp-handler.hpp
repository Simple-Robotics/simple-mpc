/// @copyright Copyright (C) 2024 INRIA
#pragma once

#include "simple-mpc/fwd.hpp"
#include "simple-mpc/ocp-handler.hpp"
#include "simple-mpc/python.hpp"

namespace simple_mpc
{
  namespace python
  {
    namespace bp = boost::python;

    namespace internal
    {
      template<typename ret_type>
      ret_type suppress_if_void(bp::detail::method_result && o)
      {
        if constexpr (!std::is_void_v<ret_type>)
        {
          return o;
        }
        else
        {
          return;
        }
      }
    } // namespace internal

#define SIMPLE_MPC_PYTHON_OVERRIDE_IMPL(ret_type, pyname, ...)                                                         \
  do                                                                                                                   \
  {                                                                                                                    \
    if (bp::override fo = this->get_override(pyname))                                                                  \
    {                                                                                                                  \
      decltype(auto) o = fo(__VA_ARGS__);                                                                              \
      return ::simple_mpc::python::internal::suppress_if_void<ret_type>(std::move(o));                                 \
    }                                                                                                                  \
  } while (false)

/**
 * @def ALIGATOR_PYTHON_OVERRIDE_PURE(ret_type, pyname, ...)
 * @brief Define the body of a virtual function override. This is meant
 *        to reduce boilerplate code when exposing virtual member functions.
 */
#define SIMPLE_MPC_PYTHON_OVERRIDE_PURE(ret_type, pyname, ...)                                                         \
  SIMPLE_MPC_PYTHON_OVERRIDE_IMPL(ret_type, pyname, __VA_ARGS__);                                                      \
  throw std::runtime_error("Tried to call pure virtual function");

/**
 * @def ALIGATOR_PYTHON_OVERRIDE(ret_type, cname, fname, ...)
 * @copybrief ALIGATOR_PYTHON_OVERRIDE_PURE()
 */
#define SIMPLE_MPC_PYTHON_OVERRIDE(ret_type, cname, fname, ...)                                                        \
  SIMPLE_MPC_PYTHON_OVERRIDE_IMPL(ret_type, #fname, __VA_ARGS__);                                                      \
  return cname::fname(__VA_ARGS__);

    template<typename T>
    inline void py_list_to_std_vector(const bp::object & iterable, std::vector<T> & out)
    {
      out = std::vector<T>(bp::stl_input_iterator<T>(iterable), bp::stl_input_iterator<T>());
    }

    template<class T>
    bp::list std_vector_to_py_list(const std::vector<T> & v)
    {
      bp::object get_iter = bp::iterator<std::vector<T>>();
      bp::object iter = get_iter(v);
      bp::list l(iter);
      return l;
    }
    struct PyOCPHandler
    : OCPHandler
    , bp::wrapper<OCPHandler>
    {
      using OCPHandler::OCPHandler;

      StageModel createStage(
        const std::map<std::string, bool> & contact_phase,
        const std::map<std::string, pinocchio::SE3> & contact_pose,
        const std::map<std::string, Eigen::VectorXd> & contact_force,
        const std::map<std::string, bool> & land_constraint) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(
          StageModel, "createStage", contact_phase, contact_pose, contact_force, land_constraint);
      }

      CostStack createTerminalCost() override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(aligator::CostStackTpl<double>, "createTerminalCost", );
      }

      void createTerminalConstraint(const Eigen::Vector3d & com_ref) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(void, "createTerminalConstraint", com_ref);
      }

      void updateTerminalConstraint(const Eigen::Vector3d & com_ref) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(void, "updateTerminalConstraint", com_ref);
      }

      void setReferencePose(const std::size_t t, const std::string & ee_name, const pinocchio::SE3 & pose_ref) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(void, "setReferencePose", t, ee_name, pose_ref);
      }

      void setReferencePoses(const std::size_t t, const std::map<std::string, pinocchio::SE3> & pose_refs) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(void, "setReferencePoses", t, pose_refs);
      }

      void setTerminalReferencePose(const std::string & ee_name, const pinocchio::SE3 & pose_ref) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(void, "setTerminalReferencePose", ee_name, pose_ref);
      }

      const pinocchio::SE3 getReferencePose(const std::size_t t, const std::string & ee_name) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(pinocchio::SE3, "getReferencePose", t, ee_name);
      }

      void setReferenceForces(const std::size_t t, const std::map<std::string, Eigen::VectorXd> & force_refs) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(void, "setReferenceForces", t, force_refs);
      }

      void
      setReferenceForce(const std::size_t t, const std::string & ee_name, const ConstVectorRef & force_ref) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(void, "setReferenceForce", t, ee_name, force_ref);
      }

      const Eigen::VectorXd getReferenceForce(const std::size_t t, const std::string & ee_name) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(Eigen::VectorXd, "getReferenceForce", t, ee_name);
      }

      void setVelocityBase(const std::size_t t, const ConstVectorRef & velocity_base) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(void, "setVelocityBase", t, velocity_base);
      }

      const Eigen::VectorXd getVelocityBase(const std::size_t t) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(Eigen::VectorXd, "getVelocityBase", t);
      }

      void setPoseBase(const std::size_t t, const ConstVectorRef & pose_base) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(void, "setPoseBase", t, pose_base);
      }

      const Eigen::VectorXd getPoseBase(const std::size_t t) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(Eigen::VectorXd, "getPoseBase", t);
      }

      const Eigen::VectorXd getProblemState(const RobotDataHandler & data_handler) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(Eigen::VectorXd, "getProblemState", data_handler);
      }

      std::size_t getContactSupport(const std::size_t t) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(std::size_t, "getContactSupport", t);
      }

      std::vector<bool> getContactState(const std::size_t t) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(std::vector<bool>, "getContactState", t);
      }

      void setReferenceState(const std::size_t t, const ConstVectorRef & x_ref) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(void, "setReferenceState", t, x_ref);
      }

      const ConstVectorRef getReferenceState(const std::size_t t) override
      {
        SIMPLE_MPC_PYTHON_OVERRIDE_PURE(ConstVectorRef, "getReferenceState", t);
      }

      void setReferenceControl(const std::size_t t, const ConstVectorRef & u_ref)
      {
        SIMPLE_MPC_PYTHON_OVERRIDE(void, OCPHandler, setReferenceControl, t, u_ref);
      }

      Eigen::VectorXd getReferenceControl(const std::size_t t)
      {
        SIMPLE_MPC_PYTHON_OVERRIDE(Eigen::VectorXd, OCPHandler, getReferenceControl, t);
      }
    };

  } // namespace python
} // namespace simple_mpc
