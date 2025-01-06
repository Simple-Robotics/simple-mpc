import numpy as np
import example_robot_data as erd
import pinocchio as pin
from bullet_robot import BulletRobot
import time
from utils import loadTalos
from simple_mpc import RobotModelHandler, RobotDataHandler, KinodynamicsOCP, MPC, IDSolver

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

nq = model_handler.getModel().nq
nv = model_handler.getModel().nv

x0 = model_handler.getReferenceState()
nu = nv - 6

""" Define kinodynamics problem """
gravity = np.array([0, 0, -9.81])
fref = np.zeros(6)
fref[2] = -model_handler.getMass() / len(model_handler.getFeetNames()) * gravity[2]
u0 = np.concatenate((fref, fref, np.zeros(nv - 6)))

w_basepos = [0, 0, 1000, 1000, 1000, 1000]
w_legpos = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
w_torsopos = [1, 1000]
w_armpos = [1, 1, 10, 10]

w_basevel = [10, 10, 10, 10, 10, 10]
w_legvel = [1, 1, 1, 1, 1, 1]
w_torsovel = [0.1, 100]
w_armvel = [10, 10, 10, 10]
w_x = np.array(
    w_basepos
    + w_legpos * 2
    + w_torsopos
    + w_armpos * 2
    + w_basevel
    + w_legvel * 2
    + w_torsovel
    + w_armvel * 2
)
w_x = np.diag(w_x) * 10
w_linforce = np.array([0.001, 0.001, 0.01])
w_angforce = np.ones(3) * 0.1
w_u = np.concatenate(
    (
        w_linforce,
        w_angforce,
        w_linforce,
        w_angforce,
        np.ones(nv - 6) * 1e-4,
    )
)
w_u = np.diag(w_u)
w_LFRF = 100000
w_cent_lin = np.array([0.0, 0.0, 1])
w_cent_ang = np.array([0.1, 0.1, 10])
w_cent = np.diag(np.concatenate((w_cent_lin, w_cent_ang)))
w_centder_lin = np.ones(3) * 0.0
w_centder_ang = np.ones(3) * 0.1
w_centder = np.diag(np.concatenate((w_centder_lin, w_centder_ang)))

problem_conf = dict(
    timestep=0.01,
    w_x=w_x,
    w_u=w_u,
    w_cent=w_cent,
    w_centder=w_centder,
    gravity=gravity,
    force_size=6,
    w_frame=np.eye(6) * w_LFRF,
    umin=-model_handler.getModel().effortLimit[6:],
    umax=model_handler.getModel().effortLimit[6:],
    qmin=model_handler.getModel().lowerPositionLimit[7:],
    qmax=model_handler.getModel().upperPositionLimit[7:],
    mu=0.8,
    Lfoot=0.1,
    Wfoot=0.075,
    kinematics_limits=True,
    force_cone=False,
)

T = 100

problem = KinodynamicsOCP(problem_conf, model_handler)
problem.createProblem(model_handler.getReferenceState(), T, 6, gravity[2], False)

""" Define MPC object """
T_ds = 20
T_ss = 80
mpc_conf = dict(
    ddpIteration=1,
    support_force=-model_handler.getMass() * gravity[2],
    TOL=1e-4,
    mu_init=1e-8,
    max_iters=1,
    num_threads=8,
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

""" Initialize whole-body inverse dynamics QP"""
contact_ids = model_handler.getFeetIds()
id_conf = dict(
    contact_ids=contact_ids,
    x0=model_handler.getReferenceState(),
    mu=0.8,
    Lfoot=0.1,
    Wfoot=0.075,
    force_size=6,
    kd=0,
    w_force=100,
    w_acc=1,
    w_tau=0,
    verbose=False,
)

qp = IDSolver(id_conf, model_handler.getModel())

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
device.changeCamera(1.0, 50, -15, [1.7, -0.5, 1.2])

q_meas, v_meas = device.measureState()
x_measured  = np.concatenate([q_meas, v_meas])

Tmpc = len(contact_phases)
nk = 2
force_size = 6

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

    start = time.time()
    mpc.iterate(x_measured)
    end = time.time()
    print("MPC iterate = " + str(end - start))

    a0 = mpc.getStateDerivative(0)[nv:].copy()
    a1 = mpc.getStateDerivative(1)[nv:].copy()
    a0[6:] = mpc.us[0][nk * force_size :]
    a1[6:] = mpc.us[1][nk * force_size :]
    forces0 = mpc.us[0][: nk * force_size]
    forces1 = mpc.us[1][: nk * force_size]
    contact_states = mpc.ocp_handler.getContactState(0)

    device.moveMarkers(
        mpc.getReferencePose(0, "left_sole_link").translation,
        mpc.getReferencePose(0, "right_sole_link").translation,
    )

    for j in range(10):
        q_meas, v_meas = device.measureState()
        x_measured  = np.concatenate([q_meas, v_meas])

        state_diff = model_handler.difference(x_measured, mpc.xs[0])
        mpc.getDataHandler().updateInternalData(x_measured, True)

        a_interp = (10 - j) / 10 * a0 + j / 10 * a1
        f_interp = (10 - j) / 10 * forces0 + j / 10 * forces1

        qp.solveQP(
            mpc.getDataHandler().getData(),
            contact_states,
            x_measured[nq:],
            a_interp,
            np.zeros(nu),
            f_interp,
            mpc.getDataHandler().getData().M,
        )

        device.execute(qp.solved_torque)
