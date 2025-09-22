import numpy as np
import pinocchio as pin
import example_robot_data as erd
from bullet_robot import BulletRobot
from simple_mpc import RobotModelHandler, RobotDataHandler, CentroidalOCP, MPC, CentroidalID, CentroidalIDSettings, Interpolator
from utils import loadTalos

# RobotWrapper
URDF_SUBPATH = "/talos_data/robots/talos_reduced.urdf"
base_joint_name ="root_joint"
reference_configuration_name = "half_sitting"

rmodelComplete, rmodel, qComplete, q0 = loadTalos()

# Create Model and Data handler
foot_points = np.array([
    [0.1, 0.075, 0],
    [-0.1, 0.075, 0],
    [-0.1, -0.075, 0],
    [0.1, -0.075, 0],
])
model_handler = RobotModelHandler(rmodel, reference_configuration_name, base_joint_name)
model_handler.addQuadFoot("left_sole_link",  base_joint_name, foot_points)
model_handler.addQuadFoot("right_sole_link", base_joint_name, foot_points)
model_handler.setFootReferencePlacement("left_sole_link", pin.XYZQUATToSE3(np.array([ 0.0, 0.1, 0.0, 0,0,0,1])))
model_handler.setFootReferencePlacement("right_sole_link", pin.XYZQUATToSE3(np.array([ 0.0,-0.1, 0.0, 0,0,0,1])))
data_handler = RobotDataHandler(model_handler)

nq = model_handler.getModel().nq
nv = model_handler.getModel().nv

x0 = np.zeros(9)
x0[:3] = data_handler.getData().com[0]
nu = model_handler.getModel().nv - 6 + len(model_handler.getFeetFrameNames()) * 6

gravity = np.array([0, 0, -9.81])
fref = np.zeros(6)
fref[2] = -model_handler.getMass() / len(model_handler.getFeetFrameNames()) * gravity[2]
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

dt_mpc = 0.01

problem_conf = dict(
    timestep=dt_mpc,
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

""" Interpolation """
N_simu = 10 # Number of substep the simulation does between two MPC computation
dt_simu = dt_mpc/N_simu
interpolator = Interpolator(model_handler.getModel())

""" Inverse Dynamics """
centroidal_ID_settings = CentroidalIDSettings()
centroidal_ID_settings.kp_base = 7.
centroidal_ID_settings.kp_com = 7.
centroidal_ID_settings.kp_posture = 10.
centroidal_ID_settings.kp_contact = 10.
centroidal_ID_settings.w_base = 50.
centroidal_ID_settings.w_com = 100.
centroidal_ID_settings.w_posture = 1.
centroidal_ID_settings.w_contact_force = 1e-6
centroidal_ID_settings.w_contact_motion = 1e-3

centroidal_ID = CentroidalID(model_handler, dt_simu, centroidal_ID_settings)

""" Initialize simulation"""
device = BulletRobot(
    model_handler.getModel().names,
    erd.getModelPath(URDF_SUBPATH),
    URDF_SUBPATH,
    dt_simu,
    model_handler.getModel(),
    model_handler.getReferenceState()[:3],
)
device.initializeJoints(model_handler.getModel().referenceConfigurations[reference_configuration_name])
device.changeCamera(1.0, 50, -15, [1.7, -0.5, 1.2])

q_meas, v_meas = device.measureState()
x_measured = np.concatenate([q_meas, v_meas])

Tmpc = len(contact_phases)
nk = 2
force_size = 6

device.showTargetToTrack(
    mpc.getDataHandler().getFootPose(mpc.getModelHandler().getFootNb("left_sole_link")),
    mpc.getDataHandler().getFootPose(mpc.getModelHandler().getFootNb("right_sole_link")),
)

v = np.zeros(6)
v[0] = 0.2
mpc.velocity_base = v
for step in range(600):
    # print("Time " + str(step))
    if step == 400:
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
    feet_ref = [mpc.getReferencePose(0, name) for name in model_handler.getFeetFrameNames()]
    feet_ref_next = [mpc.getReferencePose(1, name) for name in model_handler.getFeetFrameNames()]

    pos_com = mpc.xs[0][:3]
    pos_com_next = mpc.xs[1][:3]

    forces = mpc.us[0][: nk * force_size]
    forces_next = mpc.us[1][: nk * force_size]

    for sub_step in range(N_simu):
        t = step * dt_mpc + sub_step * dt_simu

        q_meas, v_meas = device.measureState()
        x_measured = np.concatenate([q_meas, v_meas])

        # Interpolate solution
        pos_com_interp = interpolator.interpolateLinear(sub_step, N_simu, [pos_com, pos_com_next])
        v_com = (pos_com_next - pos_com) / dt_simu

        feet_ref_interp =  [pin.SE3.Interpolate(foot_ref, foot_ref_next, (1.0 * sub_step)/N_simu) for foot_ref, foot_ref_next in zip(feet_ref, feet_ref_next)]
        feet_velocity =  [ pin.log6(foot_ref.actInv(foot_ref_next)) / dt_simu for foot_ref, foot_ref_next in zip(feet_ref, feet_ref_next)]

        forces_interp = interpolator.interpolateLinear(sub_step, N_simu, [forces, forces_next])
        forces_interp = forces_interp.reshape(2,6)
        forces_interp = [forces_interp[i, :] for i in range(2)]

        centroidal_ID.setTarget(pos_com_interp, v_com, feet_ref_interp, feet_velocity, contact_states, forces_interp)
        tau_cmd = centroidal_ID.solve(t, q_meas, v_meas)

        device.execute(tau_cmd)
