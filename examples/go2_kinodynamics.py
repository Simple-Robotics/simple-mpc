import numpy as np
from bullet_robot import BulletRobot
from simple_mpc import RobotModelHandler, RobotDataHandler, KinodynamicsOCP, MPC, IDSolver
import example_robot_data as erd
import pinocchio as pin
import time
import copy
from utils import save_trajectory

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
u0 = np.concatenate((fref, fref, fref, fref, np.zeros(model_handler.getModel().nv - 6)))


w_basepos = [0, 0, 100, 10, 10, 0]
w_legpos = [1, 1, 1]

w_basevel = [10, 10, 10, 10, 10, 10]
w_legvel = [0.1, 0.1, 0.1]
w_x = np.array(w_basepos + w_legpos * 4 + w_basevel + w_legvel * 4)
w_x = np.diag(w_x)
w_linforce = np.array([0.01, 0.01, 0.01])
w_u = np.concatenate(
    (
        w_linforce,
        w_linforce,
        w_linforce,
        w_linforce,
        np.ones(model_handler.getModel().nv - 6) * 1e-5,
    )
)
w_u = np.diag(w_u)
w_LFRF = 2000
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
    force_size=3,
    w_frame=np.eye(3) * w_LFRF,
    qmin=model_handler.getModel().lowerPositionLimit[7:],
    qmax=model_handler.getModel().upperPositionLimit[7:],
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    kinematics_limits=True,
    force_cone=False,
)
T = 50

dynproblem = KinodynamicsOCP(problem_conf, model_handler)
dynproblem.createProblem(model_handler.getReferenceState(), T, force_size, gravity[2], False)

T_ds = 10
T_ss = 30

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
    timestep=0.01,
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
contact_phases = [contact_phase_quadru] * T_ds
contact_phases += [contact_phase_lift_FL] * T_ss
contact_phases += [contact_phase_quadru] * T_ds
contact_phases += [contact_phase_lift_FR] * T_ss
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

device.initializeJoints(model_handler.getReferenceState()[:model_handler.getModel().nq])

device.changeCamera(1.0, 60, -15, [0.6, -0.2, 0.5])
nq = mpc.getModelHandler().getModel().nq
nv = mpc.getModelHandler().getModel().nv

q_meas, v_meas = device.measureState()
x_measured  = np.concatenate([q_meas, v_meas])

device.showQuadrupedFeet(
    mpc.getDataHandler().getFootPose("FL_foot"),
    mpc.getDataHandler().getFootPose("FR_foot"),
    mpc.getDataHandler().getFootPose("RL_foot"),
    mpc.getDataHandler().getFootPose("RR_foot"),
)

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
u_riccati = []
com_measured = []
solve_time = []
L_measured = []

N_simu = 10
v = np.zeros(6)
v[0] = 0.2
mpc.velocity_base = v
for t in range(300):
    # print("Time " + str(t))
    land_LF = mpc.getFootLandCycle("FL_foot")
    land_RF = mpc.getFootLandCycle("RL_foot")
    takeoff_LF = mpc.getFootTakeoffCycle("FL_foot")
    takeoff_RF = mpc.getFootTakeoffCycle("RL_foot")
    print(
        "takeoff_RF = " + str(takeoff_RF) + ", landing_RF = ",
        str(land_RF) + ", takeoff_LF = " + str(takeoff_LF) + ", landing_LF = ",
        str(land_LF),
    )
    start = time.time()
    mpc.iterate(x_measured)
    end = time.time()
    solve_time.append(end - start)

    force_FL.append(mpc.us[0][:3])
    force_FR.append(mpc.us[0][3:6])
    force_RL.append(mpc.us[0][6:9])
    force_RR.append(mpc.us[0][9:12])

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

    a0 = mpc.getStateDerivative(0)[nv:].copy()
    a1 = mpc.getStateDerivative(1)[nv:].copy()

    a0[6:] = mpc.us[0][nk * force_size :]
    a1[6:] = mpc.us[1][nk * force_size :]
    forces0 = mpc.us[0][: nk * force_size]
    forces1 = mpc.us[1][: nk * force_size]
    contact_states = mpc.ocp_handler.getContactState(0)

    device.moveQuadrupedFeet(
        mpc.getReferencePose(0, "FL_foot").translation,
        mpc.getReferencePose(0, "FR_foot").translation,
        mpc.getReferencePose(0, "RL_foot").translation,
        mpc.getReferencePose(0, "RR_foot").translation,
    )

    for j in range(N_simu):
        # time.sleep(0.01)
        q_meas, v_meas = device.measureState()
        x_measured  = np.concatenate([q_meas, v_meas])

        mpc.getDataHandler().updateInternalData(x_measured, True)

        a_interp = (N_simu - j) / N_simu * a0 + j / N_simu * a1
        f_interp = (N_simu - j) / N_simu * forces0 + j / N_simu * forces1

        qp.solveQP(
            mpc.getDataHandler().getData(),
            contact_states,
            x_measured[nq:],
            a_interp,
            np.zeros(12),
            f_interp,
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

save_trajectory(x_multibody, u_multibody, com_measured, force_FL, force_FR, force_RL, force_RR, solve_time,
                FL_measured, FR_measured, RL_measured, RR_measured,
                FL_references, FR_references, RL_references, RR_references, L_measured, "kinodynamics")
