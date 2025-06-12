import numpy as np
from bullet_robot import BulletRobot
from simple_mpc import (
    RobotModelHandler,
    RobotDataHandler,
    FullDynamicsOCP,
    MPC,
    IDSolver,
    Interpolator,
    FrictionCompensation
)
import example_robot_data as erd
import pinocchio as pin
import time
from utils import extract_forces
import copy

import matplotlib.pyplot as plt

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

nq = model_handler.getModel().nq
nv = model_handler.getModel().nv
nu = nv - 6
force_size = 3
nk = len(model_handler.getFeetNames())
nf = force_size

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
w_frame = np.eye(3)*1e3

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
    w_com = np.eye(3) * 1e3,
    umin=-model_handler.getModel().effortLimit[6:],
    umax=model_handler.getModel().effortLimit[6:],
    qmin=model_handler.getModel().lowerPositionLimit[7:],
    qmax=model_handler.getModel().upperPositionLimit[7:],
    Kp_correction=np.array([0, 0, 10]),
    Kd_correction=np.array([100, 100, 100]),
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    torque_limits=True,
    kinematics_limits=True,
    force_cone=False,
    land_cstr=True
)
T = 50

dynproblem = FullDynamicsOCP(problem_conf, model_handler)
dynproblem.createProblem(model_handler.getReferenceState(), T, force_size, gravity[2], False)

T_ds = 50
T_lift = 15
T_land = 2
T_ss = 30
N_simu = int(0.01 / 0.001)
mpc_conf = dict(
    support_force=-model_handler.getMass() * gravity[2],
    TOL=1e-4,
    mu_init=1e-8,
    max_iters=1,
    num_threads=8,
    swing_apex=0.2,
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

contact_phase_lift_Front = {
    "FL_foot": False,
    "FR_foot": False,
    "RL_foot": True,
    "RR_foot": True,
}

contact_phase_BR = {
    "FL_foot": False,
    "FR_foot": False,
    "RL_foot": False,
    "RR_foot": True,
}

contact_phase_BL = {
    "FL_foot": False,
    "FR_foot": False,
    "RL_foot": True,
    "RR_foot": False,
}


T_BR = 10
T_BL = 10
T_lift = 40

# contact_phases = [contact_phase_quadru] * T_ds
# contact_phases += [contact_phase_lift_FL] * T_ss
# contact_phases += [contact_phase_quadru] * T_ds
# contact_phases += [contact_phase_lift_FR] * T_ss

# aint that simple
contact_phases = [contact_phase_lift_Front] * T_lift
# contact_phases += [contact_phase_BR] * T_BR
# contact_phases += [contact_phase_lift_Front] * T_lift
# contact_phases += [contact_phase_BL] * T_BL

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

""" Friction """
fcompensation = FrictionCompensation(model_handler.getModel(), True)
""" Interpolation """
interpolator = Interpolator(model_handler.getModel())

""" Initialize simulation"""
device = BulletRobot(
    model_handler.getModel().names,
    erd.getModelPath(URDF_SUBPATH),
    URDF_SUBPATH,
    1e-3,
    model_handler.getModel(),
    model_handler.getReferenceState()[:3],
)

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

torques = []
torques_before_qp = []

# vitesse du robot
v = np.zeros(6)
v[1] = 0
mpc.velocity_base = v

# number of simulation steps
n_steps = 100

for t in range(n_steps):
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

    forces_vec0 = mpc.getContactForces(0)
    forces_vec1 = mpc.getContactForces(1)
    contact_states = mpc.ocp_handler.getContactState(0)

    force_FL.append(forces_vec0[:3])
    force_FR.append(forces_vec0[3:6])
    force_RL.append(forces_vec0[6:9])
    force_RR.append(forces_vec0[9:12])

    forces = [forces_vec0, forces_vec1]
    ddqs = [a0, a1]
    xss = [mpc.xs[0], mpc.xs[1]]
    uss = [mpc.us[0], mpc.us[1]]

    FL_measured.append(mpc.getDataHandler().getFootPose("FL_foot").translation.copy())
    FR_measured.append(mpc.getDataHandler().getFootPose("FR_foot").translation.copy())
    RL_measured.append(mpc.getDataHandler().getFootPose("RL_foot").translation.copy())
    RR_measured.append(mpc.getDataHandler().getFootPose("RR_foot").translation.copy())
    FL_references.append(mpc.getReferencePose(0, "FL_foot").translation.copy())
    FR_references.append(mpc.getReferencePose(0, "FR_foot").translation.copy())
    RL_references.append(mpc.getReferencePose(0, "RL_foot").translation.copy())
    RR_references.append(mpc.getReferencePose(0, "RR_foot").translation.copy())
    com_measured.append(mpc.getDataHandler().getData().com[0].copy())
    L_measured.append(mpc.getDataHandler().getData().hg.angular.copy())


    for j in range(N_simu):
        # time.sleep(0.01)
        delay = j / float(N_simu) * dt

        x_interp = interpolator.interpolateState(delay, dt, xss)
        u_interp = interpolator.interpolateLinear(delay, dt, uss)
        acc_interp = interpolator.interpolateLinear(delay, dt, ddqs)
        force_interp = interpolator.interpolateLinear(delay, dt, forces)

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
            acc_interp,
            current_torque,
            force_interp,
            mpc.getDataHandler().getData().M,
        )

        torques_before_qp.append(current_torque)

        qp_torque = qp.solved_torque.copy()

        torques.append(qp_torque)
    
        # friction_torque = fcompensation.computeFriction(x_interp[nq + 6:], current_torque)
        device.execute(current_torque)

        u_multibody.append(copy.deepcopy(current_torque))
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

torques = np.array(torques)

def plot_forces_with_bounds():
    time = np.arange(force_FL.shape[0])
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    feet = ['FL', 'FR', 'RL', 'RR']
    forces = [force_FL, force_FR, force_RL, force_RR]
    for i, (ax, f, name) in enumerate(zip(axs, forces, feet)):
        ax.plot(time, f[:, 0], label=f'{name} Fx')
        ax.plot(time, f[:, 1], label=f'{name} Fy')
        ax.plot(time, f[:, 2], label=f'{name} Fz')
        ax.set_ylabel(f'{name} Force [N]')
        ax.legend()
        ax.grid(True)
    axs[-1].set_xlabel('Time step')
    plt.tight_layout()
    plt.show()

# plot_forces_with_bounds()

n_joints = 12
n_legs = 4

torques = np.array(torques)
torques_limits = np.array(model_handler.getModel().effortLimit[6:])

with open("examples/qptorques_easy.npy", "wb") as f:
    np.save(f, torques)
    np.save(f, torques_limits)
    np.save(f, np.array(torques_before_qp))

with open("examples/com.npy", 'wb') as f:
    np.save(f, com_measured)
    np.save(f, RR_measured)
    np.save(f, RL_measured)

time = np.arange(torques.shape[0])
tau1 = np.array([t[0] for t in torques])
tau2 = np.array([t[0] for t in torques_before_qp])


plt.plot(time,tau2-tau1, label="abs(diff)")
plt.grid(True)
plt.legend()
plt.title("différence des torques avant et après QP")
plt.savefig("examples/results/diff_tau.png")
plt.show()

fig, axs = plt.subplots(2,1,figsize=(12,10), sharex=True)
axs[0].plot(time, tau2, label="tau before QP")
axs[1].plot(time, tau1, label="tau after QP")
limite = torques_limits[0]
axs[0].axhline(limite, linestyle="--")
axs[0].axhline(-limite, linestyle="--")
axs[1].axhline(limite, linestyle="--")
axs[1].axhline(-limite, linestyle="--")
axs[-1].set_xlabel('Time step')
plt.tight_layout()
plt.savefig("examples/results/tau1_tau2.png")
plt.show()

# plot_torques_with_bounds()

""" save_trajectory(x_multibody, u_multibody, com_measured, force_FL, force_FR, force_RL, force_RR, solve_time,
                FL_measured, FR_measured, RL_measured, RR_measured,
                FL_references, FR_references, RL_references, RR_references, L_measured, "fulldynamics") """
