import numpy as np
import pinocchio as pin
import example_robot_data as erd
from bullet_robot import BulletRobot
from simple_mpc import RobotModelHandler, RobotDataHandler, CentroidalOCP, MPC, IKIDSolver
from utils import loadTalos
import time

# RobotWrapper
URDF_SUBPATH = "/talos_data/robots/talos_reduced.urdf"
base_joint_name ="root_joint"
reference_configuration_name = "half_sitting"

rmodelComplete, rmodel, qComplete, q0 = loadTalos()

# Create Model and Data handler
model_handler = RobotModelHandler(rmodel, reference_configuration_name, base_joint_name)
model_handler.addFoot("left_sole_link",  base_joint_name, pin.XYZQUATToSE3(np.array([ 0.0, 0.1, 0.0, 0,0,0,1])))
model_handler.addFoot("right_sole_link", base_joint_name, pin.XYZQUATToSE3(np.array([ 0.0,-0.1, 0.0, 0,0,0,1])))
data_handler = RobotDataHandler(model_handler)

x0 = np.zeros(9)
x0[:3] = data_handler.getData().com[0]
nu = model_handler.getModel().nv - 6 + len(model_handler.getFeetNames()) * 6

gravity = np.array([0, 0, -9.81])
fref = np.zeros(6)
fref[2] = -model_handler.getMass() / len(model_handler.getFeetNames()) * gravity[2]
u0 = np.concatenate((fref, fref))

w_control_linear = np.ones(3) * 0.001
w_control_angular = np.ones(3) * 0.1
w_u = np.diag(
    np.concatenate(
        (w_control_linear, w_control_angular, w_control_linear, w_control_angular)
    )
)
w_com = np.diag(np.array([0, 0, 0]))
w_linear_mom = np.diag(np.array([0.01, 0.01, 100]))
w_linear_acc = 0.01 * np.eye(3)
w_angular_mom = np.diag(np.array([0.1, 0.1, 1000]))
w_angular_acc = 0.01 * np.eye(3)

problem_conf = dict(
    timestep=0.01,
    w_u=w_u,
    w_com=w_com,
    w_linear_mom=w_linear_mom,
    w_angular_mom=w_angular_mom,
    w_linear_acc=w_linear_acc,
    w_angular_acc=w_angular_acc,
    gravity=gravity,
    mu=0.8,
    Lfoot=0.1,
    Wfoot=0.075,
    force_size=6,
)
T = 100

problem = CentroidalOCP(problem_conf, model_handler)
problem.createProblem(data_handler.getCentroidalState(), T, 6, gravity[2], False)

T_ds = 20
T_ss = 80

mpc_conf = dict(
    ddpIteration=1,
    support_force=-model_handler.getMass() * gravity[2],
    TOL=1e-4,
    mu_init=1e-8,
    max_iters=1,
    num_threads=1,
    swing_apex=0.15,
    T_fly=T_ss,
    T_contact=T_ds,
    timestep=problem_conf["timestep"],
)

mpc = MPC(mpc_conf, problem)

""" Define contact sequence throughout horizon"""
contact_phase_double = {
    "left_sole_link": True,
    "right_sole_link": True,
}
contact_phase_left = {
    "left_sole_link": True,
    "right_sole_link": False,
}
contact_phase_right = {
    "left_sole_link": False,
    "right_sole_link": True,
}

contact_phases = [contact_phase_double] * T_ds
contact_phases += [contact_phase_left] * T_ss
contact_phases += [contact_phase_double] * T_ds
contact_phases += [contact_phase_right] * T_ss

mpc.generateCycleHorizon(contact_phases)

""" Initialize inverse dynamics QP """
g_basepos = [0, 0, 0, 10, 10, 10]
g_legpos = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
g_torsopos = [1, 1]
g_armpos = [10, 10, 100, 10]

g_q = np.array(g_basepos + g_legpos * 2 + g_torsopos + g_armpos * 2) * 10

g_p = np.array([2000, 2000, 2000, 2000, 2000, 2000])
g_b = np.array([10, 10, 10])

Kp_gains = [g_q, g_p, g_b]
Kd_gains = [2 * np.sqrt(g_q), 2 * np.sqrt(g_p), 2 * np.sqrt(g_b)]
contact_ids = model_handler.getFeetIds()
fixed_frame_ids = [model_handler.getBaseFrameId(), model_handler.getModel().getFrameId("torso_2_link")]
ikid_conf = dict(
    Kp_gains=Kp_gains,
    Kd_gains=Kd_gains,
    contact_ids=contact_ids,
    fixed_frame_ids=fixed_frame_ids,
    x0=model_handler.getReferenceState(),
    dt=0.01,
    mu=0.8,
    Lfoot=0.1,
    Wfoot=0.075,
    force_size=6,
    w_qref=500,
    w_footpose=50000,
    w_centroidal=10,
    w_baserot=1000,
    w_force=100,
    verbose=False,
)

qp = IKIDSolver(ikid_conf, model_handler.getModel())

""" Initialize simulation"""
device = BulletRobot(
    model_handler.getControlledJointNames(),
    erd.getModelPath(URDF_SUBPATH),
    URDF_SUBPATH,
    1e-3,
    model_handler.getModel(),
    model_handler.getReferenceState()[:3],
)
device.initializeJoints(model_handler.getModel().referenceConfigurations[reference_configuration_name])
device.changeCamera(1.0, 90, -5, [1.5, 0, 1])

nq = mpc.getModelHandler().getModel().nq
nv = mpc.getModelHandler().getModel().nv

q_meas, v_meas = device.measureState()
x_measured = np.concatenate([q_meas, v_meas])

Tmpc = len(contact_phases)
nk = 2
force_size = 6
x_centroidal = mpc.getDataHandler().getCentroidalState()

device.showTargetToTrack(
    mpc.getDataHandler().getFootPose("left_sole_link"),
    mpc.getDataHandler().getFootPose("right_sole_link"),
)

v = np.zeros(6)
v[0] = 0.2
mpc.velocity_base = v
for t in range(600):
    # print("Time " + str(t))
    if t == 400:
        print("SWITCH TO STAND")
        mpc.switchToStand()

    land_LF = mpc.getFootLandCycle("left_sole_link")
    land_RF = mpc.getFootLandCycle("right_sole_link")
    takeoff_LF = mpc.getFootTakeoffCycle("left_sole_link")
    takeoff_RF = mpc.getFootTakeoffCycle("right_sole_link")

    print(
        "takeoff_RF = " + str(takeoff_RF) + ", landing_RF = ",
        str(land_RF) + ", takeoff_LF = " + str(takeoff_LF) + ", landing_LF = ",
        str(land_LF),
    )

    mpc.iterate(x_measured)

    device.moveMarkers(
        mpc.getReferencePose(0, "left_sole_link").translation,
        mpc.getReferencePose(0, "right_sole_link").translation,
    )

    contact_states = mpc.ocp_handler.getContactState(0)
    foot_ref = [mpc.getReferencePose(0, name) for name in model_handler.getFeetNames()]
    foot_ref_next = [mpc.getReferencePose(1, name) for name in model_handler.getFeetNames()]
    dH = mpc.getStateDerivative(0)[3:9]
    qp.computeDifferences(
        mpc.getDataHandler().getData(), x_measured, foot_ref, foot_ref_next
    )

    for j in range(10):
        time.sleep(0.001)
        q_meas, v_meas = device.measureState()
        x_measured = np.concatenate([q_meas, v_meas])

        mpc.getDataHandler().updateInternalData(x_measured, True)
        x_centroidal = mpc.getDataHandler().getCentroidalState()
        state_diff = mpc.xs[0] - x_centroidal

        forces = (
            mpc.us[0][: nk * force_size]
            - 1 * mpc.solver.results.controlFeedbacks()[0] @ state_diff
        )


        qp.solve_qp(
            mpc.getDataHandler().getData(),
            contact_states,
            x_measured[nq:],
            forces,
            dH,
            mpc.getDataHandler().getData().M,
        )


        device.execute(qp.solved_torque)
