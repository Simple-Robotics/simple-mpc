/// @copyright Copyright (C) 2024 INRIA
#include "simple-mpc/config.hpp"
#include "simple-mpc/fwd.hpp"
#include "simple-mpc/python.hpp"

namespace simple_mpc::python
{

  namespace bp = boost::python;

  /* FORWARD DECLARATIONS */
  void exposeHandler();
  void exposeOcpHandler();
  void exposeFullDynamicsOcp();
  void exposeCentroidalOcp();
  void exposeKinodynamicsOcp();
  void exposeMPC();
  void exposeInterpolator();
  void exposeFrictionCompensation();
  void exposeInverseDynamics();

  /* PYTHON MODULE */
  BOOST_PYTHON_MODULE(simple_mpc_pywrap)
  {
    bp::import("pinocchio");
    bp::import("aligator");
    bp::scope().attr("__version__") = SIMPLE_MPC_VERSION;
    ENABLE_SPECIFIC_MATRIX_TYPE(Eigen::VectorXi);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector6d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector7d);
    exposeHandler();
    exposeOcpHandler();
    exposeFullDynamicsOcp();
    exposeCentroidalOcp();
    exposeKinodynamicsOcp();
    exposeMPC();
    exposeInterpolator();
    exposeInverseDynamics();
    exposeFrictionCompensation();
  }

} // namespace simple_mpc::python
