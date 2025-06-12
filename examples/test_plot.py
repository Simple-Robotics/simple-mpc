
import numpy as np
import matplotlib.pyplot as plt

n_joints = 12
n_legs = 4
n_steps = 100

T_ds = 30
T_lift = 10
T_land = 2
T_ss = 30

T = (T_ds/2, T_lift, T_ss, T_land, T_ds/2, T_ds/2, T_lift, T_ss, T_land, T_ds/2)
c0 = np.zeros(len(T) + 1)
for i in range(len(T)):
    c0[i] = sum(T[:i])

c0 = 500 + 10*c0

with open('examples/qptorques.npy', 'rb') as f:
    torques = np.load(f)
    torques_limits = np.load(f)
    # torques_beforeQP = np.load(f)

contact_changes = np.zeros(torques.shape[0])

joint_names = ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint",
                "FR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]

def plot_torques_with_bounds():
    # TODO: plot torques for each leg in a seperate subfig
    fig, axs = plt.subplots(n_legs,1, figsize=(12,10), sharex=True)    # 4 legs

    # simulation step of 1 ms
    time = np.arange(torques.shape[0])

    # joint torques colors
    colors = ["orange", "red", "blue"]

    for t, ax in enumerate(axs):  # tous les couples
        for i in c0:
            ax.axvline(i, color='teal')     # steps of jumps...

        for j, c in enumerate(colors):
            t1 = [x[3*t + j] for x in torques]
            ax.plot(time, t1, label=f"{joint_names[3*t + j]}", color=c)
            ax.axhline(torques_limits[3*t+j], color=c, linestyle='--')
            ax.axhline(-torques_limits[3*t+j], color=c, linestyle='--')

        ax.set_ylabel("Torque [N.m]")

        ax.legend()
        ax.grid(True)
    axs[-1].set_xlabel('Time (ms)')
    plt.suptitle(r"Joint torques per leg with bounds ($\tau$)", fontsize=16)
    plt.tight_layout()
    plt.savefig("examples/results/torques_qp.png")
    plt.show()


def plot_acc():
    # TODO: check if accelerations are ok
    pass


def compare_torques():
    time = np.arange(torques_beforeQP.shape[0])
    tau1 = np.array([t[0] for t in torques])
    tau2 = np.array([t[0] for t in torques_beforeQP])

    fig, axs = plt.subplots(2,1,figsize=(12,10), sharex=True)
    axs[0].plot(time, tau2, label="tau before QP")
    axs[1].plot(time, tau1, label="tau after QP")
    limite = torques_limits[0]
    axs[0].axhline(limite, linestyle="--")
    axs[0].axhline(-limite, linestyle="--")
    axs[1].axhline(limite, linestyle="--")
    axs[1].axhline(-limite, linestyle="--")
    axs[-1].set_xlabel('Time step')
    axs[0].grid(True)
    axs[1].grid(True)
    axs[0].legend()
    axs[1].legend()
    plt.suptitle(r"Joint torque ($\tau$) of FL hip joint before and after QP", fontsize=16)
    plt.tight_layout()
    plt.savefig("examples/results/Tau1AndTau2.png")
    plt.show()

# compare_torques()

plot_torques_with_bounds()