import numpy as np
import pinocchio as pin
import example_robot_data
import os

CURRENT_DIRECTORY = os.getcwd()
DEFAULT_SAVE_DIR = CURRENT_DIRECTORY + "/tmp"


def loadTalos():
    robotComplete = example_robot_data.load("talos")
    qComplete = robotComplete.model.referenceConfigurations["half_sitting"]

    locked_joints_names = [
        "arm_left_5_joint",
        "arm_left_6_joint",
        "arm_left_7_joint",
        "gripper_left_joint",
        "arm_right_5_joint",
        "arm_right_6_joint",
        "arm_right_7_joint",
        "gripper_right_joint",
        "head_1_joint",
        "head_2_joint",
    ]
    locked_joints = [robotComplete.model.getJointId(el) for el in locked_joints_names]
    robot = robotComplete.buildReducedRobot(locked_joints, qComplete)
    rmodel: pin.Model = robot.model
    q0 = rmodel.referenceConfigurations["half_sitting"]

    return robotComplete.model, rmodel, qComplete, q0


def save_trajectory(
    xs,
    us,
    # uq,
    com,
    FL_force,
    FR_force,
    RL_force,
    RR_force,
    solve_time,
    FL_trans,
    FR_trans,
    RL_trans,
    RR_trans,
    FL_trans_ref,
    FR_trans_ref,
    RL_trans_ref,
    RR_trans_ref,
    L_measured,
    save_name=None,
    save_dir=DEFAULT_SAVE_DIR,
):
    """
    Saves data to a compressed npz file (binary)
    """
    simu_data = {}
    simu_data["xs"] = xs
    simu_data["us"] = us
    # simu_data["uq"] = uq
    simu_data["com"] = com
    simu_data["FL_force"] = FL_force
    simu_data["FR_force"] = FR_force
    simu_data["RL_force"] = RL_force
    simu_data["RR_force"] = RR_force
    simu_data["FL_trans"] = FL_trans
    simu_data["FR_trans"] = FR_trans
    simu_data["RL_trans"] = RL_trans
    simu_data["RR_trans"] = RR_trans
    simu_data["FL_trans_ref"] = FL_trans_ref
    simu_data["FR_trans_ref"] = FR_trans_ref
    simu_data["RL_trans_ref"] = RL_trans_ref
    simu_data["RR_trans_ref"] = RR_trans_ref
    simu_data["L_measured"] = L_measured
    simu_data["solve_time"] = solve_time
    print("Compressing & saving data...")
    if save_name is None:
        save_name = "sim_data_NO_NAME"
    if save_dir is None:
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    save_path = save_dir + "/" + save_name + ".npz"
    np.savez_compressed(save_path, data=simu_data)
    print("Saved data to " + str(save_path) + " !")


def load_data(npz_file):
    """
    Loads a npz archive of sim_data into a dict
    """
    d = np.load(npz_file, allow_pickle=True, encoding="latin1")
    return d["data"][()]


def extract_forces(problem, workspace, id):
    force_FL = np.zeros(3)
    force_FR = np.zeros(3)
    force_RL = np.zeros(3)
    force_RR = np.zeros(3)
    in_contact = problem.stages[
        id
    ].dynamics.differential_dynamics.constraint_models.__len__()
    for i in range(in_contact):
        if (
            problem.stages[id].dynamics.differential_dynamics.constraint_models[i].name
            == "FL_foot"
        ):
            force_FL = (
                workspace.problem_data.stage_data[id]
                .dynamics_data.continuous_data.constraint_datas[i]
                .contact_force.linear
            )
        elif (
            problem.stages[id].dynamics.differential_dynamics.constraint_models[i].name
            == "FR_foot"
        ):
            force_FR = (
                workspace.problem_data.stage_data[id]
                .dynamics_data.continuous_data.constraint_datas[i]
                .contact_force.linear
            )
        elif (
            problem.stages[id].dynamics.differential_dynamics.constraint_models[i].name
            == "RL_foot"
        ):
            force_RL = (
                workspace.problem_data.stage_data[id]
                .dynamics_data.continuous_data.constraint_datas[i]
                .contact_force.linear
            )
        elif (
            problem.stages[id].dynamics.differential_dynamics.constraint_models[i].name
            == "RR_foot"
        ):
            force_RR = (
                workspace.problem_data.stage_data[id]
                .dynamics_data.continuous_data.constraint_datas[i]
                .contact_force.linear
            )

    return force_FL, force_FR, force_RL, force_RR
