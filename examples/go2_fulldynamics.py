import numpy as np
from bullet_robot import BulletRobot
from simple_mpc import RobotModelHandler, RobotDataHandler, FullDynamicsOCP, MPC, IDSolver
import example_robot_data as erd
import pinocchio as pin
import time
from utils import extract_forces
import copy

# ####### CONFIGURATION  ############
# Load robot
URDF_SUBPATH = "/go2_description/urdf/go2.urdf"
base_joint_name ="root_joint"
robot_wrapper = erd.load('go2')

# Create Model and Data handler
model_handler = RobotModelHandler(robot_wrapper.model, "standing", base_joint_name)
model_handler.addFoot("FL_foot", base_joint_name, pin.XYZQUATToSE3(np.array([ 0.17, 0.15, 0.0, 0,0,0,1])))
model_handler.addFoot("FR_foot", base_joint_name, pin.XYZQUATToSE3(np.array([ 0.17,-0.15, 0.0, 0,0,0,1])))
model_handler.addFoot("RL_foot", base_joint_name, pin.XYZQUATToSE3(np.array([-0.24, 0.15, 0.0, 0,0,0,1])))
model_handler.addFoot("RR_foot", base_joint_name, pin.XYZQUATToSE3(np.array([-0.24,-0.15, 0.0, 0,0,0,1])))
data_handler = RobotDataHandler(model_handler)

force_size = 3
nk = len(model_handler.getFeetNames())
gravity = np.array([0, 0, -9.81])
fref = np.zeros(force_size)
fref[2] = -model_handler.getMass() / nk * gravity[2]
u0 = np.zeros(model_handler.getModel().nv - 6)

w_basepos = [0, 0, 0, 0, 0, 0]
w_legpos = [10, 10, 10]

w_basevel = [10, 10, 10, 10, 10, 10]
w_legvel = [0.1, 0.1, 0.1]
w_x = np.array(w_basepos + w_legpos * 4 + w_basevel + w_legvel * 4)
w_cent_lin = np.array([0.0, 0.0, 0])
w_cent_ang = np.array([0, 0, 0])
w_forces_lin = np.array([0.0001, 0.0001, 0.0001])
w_frame = np.diag(np.array([1000, 1000, 1000]))

dt = 0.01
problem_conf = dict(
    timestep=dt,
    w_x=np.diag(w_x),
    w_u=np.eye(u0.size) * 1e-4,
    w_cent=np.diag(np.concatenate((w_cent_lin, w_cent_ang))),
    gravity=gravity,
    force_size=3,
    w_forces=np.diag(w_forces_lin),
    w_frame=w_frame,
    umin=-model_handler.getModel().effortLimit[6:],
    umax=model_handler.getModel().effortLimit[6:],
    qmin=model_handler.getModel().lowerPositionLimit[7:],
    qmax=model_handler.getModel().upperPositionLimit[7:],
    Kp_correction=np.array([0, 0, 0]),
    Kd_correction=np.array([0, 0, 0]),
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    torque_limits=True,
    kinematics_limits=True,
    force_cone=False,
)
T = 50

dynproblem = FullDynamicsOCP(problem_conf, model_handler)
dynproblem.createProblem(model_handler.getReferenceState(), T, force_size, gravity[2], False)

T_ds = 10
T_ss = 30
N_simu = int(0.01 / 0.001)
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
    timestep=dt,
)

mpc = MPC(mpc_conf, dynproblem)

""" Define contact sequence throughout horizon"""
contact_phase_quadru = {
    "FL_foot": True,
    "FR_foot": True,
    "RL_foot": True,
    "RR_foot": True,
}
contact_phase_lift_FL = {
    "FL_foot": False,
    "FR_foot": True,
    "RL_foot": True,
    "RR_foot": False,
}
contact_phase_lift_FR = {
    "FL_foot": True,
    "FR_foot": False,
    "RL_foot": False,
    "RR_foot": True,
}
contact_phase_lift = {
    "FL_foot": False,
    "FR_foot": False,
    "RL_foot": False,
    "RR_foot": False,
}
contact_phases = [contact_phase_quadru] * int(T_ds / 2)
contact_phases += [contact_phase_lift_FL] * T_ss
contact_phases += [contact_phase_quadru] * T_ds
contact_phases += [contact_phase_lift_FR] * T_ss
contact_phases += [contact_phase_quadru] * int(T_ds / 2)

""" contact_phases = [contact_phase_quadru] * int(T_ds / 2)
contact_phases += [contact_phase_lift] * T_ss
contact_phases += [contact_phase_quadru] * int(T_ds / 2) """

mpc.generateCycleHorizon(contact_phases)

""" Initialize whole-body inverse dynamics QP"""
contact_ids = model_handler.getFeetIds()
id_conf = dict(
    contact_ids=contact_ids,
    x0=model_handler.getReferenceState(),
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    force_size=3,
    kd=0,
    w_force=0,
    w_acc=0,
    w_tau=1,
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

nq = mpc.getModelHandler().getModel().nq
nv = mpc.getModelHandler().getModel().nv
device.initializeJoints(model_handler.getReferenceState()[:nq])

for i in range(40):
    device.setFrictionCoefficients(i, 10, 0)
#device.changeCamera(1.0, 60, -15, [0.6, -0.2, 0.5])

q_meas, v_meas = device.measureState()
x_measured  = np.concatenate([q_meas, v_meas])
mpc.getDataHandler().updateInternalData(x_measured, False)

ref_foot_pose = [mpc.getDataHandler().getRefFootPose(mpc.getModelHandler().getFeetNames()[i]) for i in range(4)]
for pose in ref_foot_pose:
    pose.translation[2] = 0
device.showQuadrupedFeet(*ref_foot_pose)
Tmpc = len(contact_phases)

force_FL = []
force_FR = []
force_RL = []
force_RR = []
FL_measured = []
FR_measured = []
RL_measured = []
RR_measured = []
FL_references = []
FR_references = []
RL_references = []
RR_references = []
x_multibody = []
u_multibody = []
com_measured = []
solve_time = []
L_measured = []

v = np.zeros(6)
v[0] = 0.2
mpc.velocity_base = v
for t in range(500):
    print("Time " + str(t))
    land_LF = mpc.getFootLandCycle("FL_foot")
    land_RF = mpc.getFootLandCycle("RL_foot")
    takeoff_LF = mpc.getFootTakeoffCycle("FL_foot")
    takeoff_RF = mpc.getFootTakeoffCycle("RL_foot")
    """ print(
        "takeoff_RF = " + str(takeoff_RF) + ", landing_RF = ",
        str(land_RF) + ", takeoff_LF = " + str(takeoff_LF) + ", landing_LF = ",
        str(land_LF),
    ) """

    """ if t == 200:
        for s in range(T):
            device.resetState(mpc.xs[s][:nq])
            #device.resetState(state_ref[s])
            time.sleep(0.02)
            print("s = " + str(s))
        exit()  """

    device.moveQuadrupedFeet(
        mpc.getReferencePose(0, "FL_foot").translation,
        mpc.getReferencePose(0, "FR_foot").translation,
        mpc.getReferencePose(0, "RL_foot").translation,
        mpc.getReferencePose(0, "RR_foot").translation,
    )

    start = time.time()
    mpc.iterate(x_measured)
    end = time.time()
    solve_time.append(end - start)

    a0 = mpc.getStateDerivative(0)[nv:]
    a1 = mpc.getStateDerivative(1)[nv:]

    FL_f, FR_f, RL_f, RR_f = extract_forces(mpc.getTrajOptProblem(), mpc.solver.workspace, 0)
    contact_states = mpc.ocp_handler.getContactState(0)
    total_forces = np.concatenate((FL_f, FR_f, RL_f, RR_f))
    force_FL.append(FL_f)
    force_FR.append(FR_f)
    force_RL.append(RL_f)
    force_RR.append(RR_f)

    FL_measured.append(mpc.getDataHandler().getFootPose("FL_foot").translation)
    FR_measured.append(mpc.getDataHandler().getFootPose("FR_foot").translation)
    RL_measured.append(mpc.getDataHandler().getFootPose("RL_foot").translation)
    RR_measured.append(mpc.getDataHandler().getFootPose("RR_foot").translation)
    FL_references.append(mpc.getReferencePose(0, "FL_foot").translation)
    FR_references.append(mpc.getReferencePose(0, "FR_foot").translation)
    RL_references.append(mpc.getReferencePose(0, "RL_foot").translation)
    RR_references.append(mpc.getReferencePose(0, "RR_foot").translation)
    com_measured.append(mpc.getDataHandler().getData().com[0].copy())
    L_measured.append(mpc.getDataHandler().getData().hg.angular.copy())


    for j in range(N_simu):
        # time.sleep(0.01)
        u_interp = (N_simu - j) / N_simu * mpc.us[0] + j / N_simu * mpc.us[1]
        a_interp = (N_simu - j) / N_simu * a0 + j / N_simu * a1
        x_interp = (N_simu - j) / N_simu * mpc.xs[0] + j / N_simu * mpc.xs[1]
        K_interp = (N_simu - j) / N_simu * mpc.Ks[0] + j / N_simu * mpc.Ks[1]

        q_meas, v_meas = device.measureState()
        x_measured = np.concatenate([q_meas, v_meas])

        mpc.getDataHandler().updateInternalData(x_measured, True)

        current_torque = u_interp - 1. * mpc.Ks[0] @ model_handler.difference(
            x_measured, x_interp
        )

        qp.solveQP(
            mpc.getDataHandler().getData(),
            contact_states,
            x_measured[nq:],
            a0,
            current_torque,
            total_forces,
            mpc.getDataHandler().getData().M,
        )

        device.execute(qp.solved_torque)

        u_multibody.append(copy.deepcopy(qp.solved_torque))
        x_multibody.append(x_measured)

force_FL = np.array(force_FL)
force_FR = np.array(force_FR)
force_RL = np.array(force_RL)
force_RR = np.array(force_RR)
solve_time = np.array(solve_time)
FL_measured = np.array(FL_measured)
FR_measured = np.array(FR_measured)
RL_measured = np.array(RL_measured)
RR_measured = np.array(RR_measured)
FL_references = np.array(FL_references)
FR_references = np.array(FR_references)
RL_references = np.array(RL_references)
RR_references = np.array(RR_references)
com_measured = np.array(com_measured)
L_measured = np.array(L_measured)

""" save_trajectory(x_multibody, u_multibody, com_measured, force_FL, force_FR, force_RL, force_RR, solve_time,
                FL_measured, FR_measured, RL_measured, RR_measured,
                FL_references, FR_references, RL_references, RR_references, L_measured, "fulldynamics") """
