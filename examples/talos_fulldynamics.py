"""
This script launches a locomotion MPC scheme which solves repeatedly an
optimal control problem based on the full dynamics model of the humanoid robot Talos.
The contacts forces are modeled as 6D wrenches.
"""

import numpy as np
import time
from bullet_robot import BulletRobot
import pinocchio as pin
from simple_mpc import MPC, FullDynamicsOCP, RobotModelHandler, RobotDataHandler
from utils import loadTalos
import example_robot_data as erd

# ####### CONFIGURATION  ############
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

controlled_joints = rmodel.names[1:].tolist()
controlled_ids = [rmodelComplete.getJointId(name_joint) for name_joint in controlled_joints[1:]]

nq = model_handler.getModel().nq
nv = model_handler.getModel().nv
nu = nv - 6

x0 = model_handler.getReferenceState()
nu = model_handler.getModel().nv - 6
w_basepos = [0, 0, 0, 10, 10, 10]
w_legpos = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
w_torsopos = [1, 100]
w_armpos = [1, 1, 10, 10]

w_basevel = [10, 10, 10, 10, 10, 10]
w_legvel = [1, 1, 1, 1, 1, 1]
w_torsovel = [1, 100]
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
w_cent_lin = np.array([0.1, 0.1, 10])
w_cent_ang = np.array([0.1, 0.1, 10])
w_forces_lin = np.array([0.001, 0.001, 0.001])
w_forces_ang = np.ones(3) * 0.001
gravity = np.array([0, 0, -9.81])

problem_conf = dict(
    timestep=0.01,
    w_x=np.diag(w_x),
    w_u=np.eye(nu) * 1e-4,
    w_cent=np.diag(np.concatenate((w_cent_lin, w_cent_ang))),
    gravity=gravity,
    force_size=6,
    w_forces=np.diag(np.concatenate((w_forces_lin, w_forces_ang))),
    w_frame=np.eye(6) * 2000,
    umin=-model_handler.getModel().effortLimit[6:],
    umax=model_handler.getModel().effortLimit[6:],
    qmin=model_handler.getModel().lowerPositionLimit[7:],
    qmax=model_handler.getModel().upperPositionLimit[7:],
    Kp_correction=np.array([0, 0, 50, 0, 0, 0]),
    Kd_correction=np.array([100, 100, 100, 100, 100 ,100]),
    mu=0.8,
    Lfoot=0.1,
    Wfoot=0.075,
    torque_limits=True,
    kinematics_limits=True,
    force_cone=True,
)

T = 100
dynproblem = FullDynamicsOCP(problem_conf, model_handler)
dynproblem.createProblem(x0, T, 6, gravity[2], False)

""" Define feet trajectory """
T_ss = 80
T_ds = 20
totalSteps = 1
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

mpc = MPC(mpc_conf, dynproblem)

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

Tmpc = len(contact_phases)
mpc.generateCycleHorizon(contact_phases)
problem = mpc.getTrajOptProblem()

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

q_meas, v_meas = device.measureState()
x_measured = np.concatenate((q_meas, v_meas))

land_LF = -1
land_RF = -1
takeoff_LF = -1
takeoff_RF = -1
device.showTargetToTrack(
    mpc.getDataHandler().getFootPose("left_sole_link"),
    mpc.getDataHandler().getFootPose("right_sole_link"),
)

v = np.zeros(6)
v[0] = 0.1
mpc.velocity_base = v
for t in range(Tmpc + 800):
    # print("Time " + str(t))
    land_LF = mpc.getFootLandCycle("left_sole_link")
    land_RF = mpc.getFootLandCycle("right_sole_link")
    takeoff_LF = mpc.getFootTakeoffCycle("left_sole_link")
    takeoff_RF = mpc.getFootTakeoffCycle("right_sole_link")

    if t == 600:
        print("SWITCH TO STAND")
        mpc.switchToStand()

    print(
        "takeoff_RF = " + str(takeoff_RF) + ", landing_RF = ",
        str(land_RF) + ", takeoff_LF = " + str(takeoff_LF) + ", landing_LF = ",
        str(land_LF),
    )
    start = time.time()

    mpc.iterate(x_measured)
    end = time.time()
    print("MPC iterate = " + str(end - start))
    device.moveMarkers(
        mpc.getReferencePose(0, "left_sole_link").translation,
        mpc.getReferencePose(0, "right_sole_link").translation,
    )

    for j in range(10):
        q_meas, v_meas = device.measureState()
        x_measured = np.concatenate((q_meas, v_meas))

        current_torque = mpc.us[0] - mpc.Ks[0] @ model_handler.difference(x_measured, mpc.xs[0])
        device.execute(current_torque)
