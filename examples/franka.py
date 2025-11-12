import numpy as np
from simple_mpc import RobotModelHandler, RobotDataHandler, ArmDynamicsOCP, ArmMPC
import pinocchio as pin
import coal
import example_robot_data as erd
from pinocchio.visualize import MeshcatVisualizer
import time
import simple

# ####### CONFIGURATION  ############
# Load robot
robotComplete = erd.load("panda")
rmodelComplete: pin.Model = robotComplete.model

# Reduce model
qComplete = rmodelComplete.referenceConfigurations["default"]
locked_joints = [8, 9]

robot = robotComplete.buildReducedRobot(locked_joints, qComplete)
rmodel: pin.Model = robot.model
rdata = rmodel.createData()

geom_model = robot.collision_model
q0 = rmodel.referenceConfigurations["default"]
tool_name = "panda_leftfinger"
tool_id = rmodel.getFrameId(tool_name)
visual_model = robot.visual_model

# Create the simulator object
simulator = simple.Simulator(rmodel, geom_model)

# Create Model and Data handler
model_handler = RobotModelHandler(rmodel, "default", "panda_link0")
data_handler = RobotDataHandler(model_handler)

# Create OCP
w_q = np.ones(7) * 1
w_v = np.ones(7) * 1
w_x = np.concatenate((w_q, w_v))
w_u = np.ones(7) * 1e-2
w_frame = np.diag(np.array([1000, 1000, 1000, 10, 10, 10]))

dt = 0.01
dt_sim = 1e-3
problem_conf = dict(
    timestep=dt,
    w_x=np.diag(w_x),
    w_u=np.diag(w_u),
    gravity=np.array([0, 0, -9.81]),
    w_frame=w_frame,
    umin=-model_handler.getModel().effortLimit,
    umax=model_handler.getModel().effortLimit,
    qmin=model_handler.getModel().lowerPositionLimit,
    qmax=model_handler.getModel().upperPositionLimit,
    torque_limits=True,
    kinematics_limits=True,
    ee_name=tool_name,
)
T = 100

dynproblem = ArmDynamicsOCP(problem_conf, model_handler)
dynproblem.createProblem(model_handler.getReferenceState(), T)

# Create MPC
N_simu = int(dt / dt_sim)
mpc_conf = dict(
    TOL=1e-4,
    mu_init=1e-8,
    max_iters=1,
    num_threads=1,
    timestep=dt,
)

mpc = ArmMPC(mpc_conf, dynproblem)

target = pin.SE3.Identity()
target.rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
target.translation = np.array([0.15, 0.5, 0.5])

nv = mpc.getModelHandler().getModel().nv
nx = nv * 2

q = q0.copy()
v = np.zeros(nv)
x_measured = np.concatenate([q, v])
mpc.getDataHandler().updateInternalData(x_measured, False)

# Visualization of target
fr_name = "universe"
fr_id = rmodel.getFrameId(fr_name)
joint_id = rmodel.frames[fr_id].parentJoint
target_place = pin.SE3.Identity()
target_place.translation = target.translation
target_object1 = pin.GeometryObject(
    "target1", fr_id, joint_id, coal.Sphere(0.02), target_place
)
target_object1.meshColor[:] = [0.5, 0.5, 1.0, 1.0]
visual_model.addGeometryObject(target_object1)
target_id1 = visual_model.getGeometryId("target1")
visual_data = visual_model.createData()

z_mov = 0.2
x_mov = 0.2
freq_mov = 1

### Load visualizer
vizer = MeshcatVisualizer(rmodel, geom_model, visual_model, data=rdata)
vizer.initViewer(open=True, loadModel=True)
vizer.display(pin.neutral(rmodel))
vizer.setBackgroundColor()

vizer.display(q)

Tmpc = 1000
target_new = target.copy()

total = 0
print("Start simu")
for t in range(Tmpc):
    if t == 300:
        # Stop tracking target
        print("Switch to rest")
        mpc.switchToRest()
    if t > 600 or t < 300:
        # Track target
        mpc.switchToReach(target_new)
    print("Time " + str(t))
    for j in range(N_simu):
        u_interp = (N_simu - j) / N_simu * mpc.us[0] + j / N_simu * mpc.us[1]
        x_interp = (N_simu - j) / N_simu * mpc.xs[0] + j / N_simu * mpc.xs[1]

        mpc.getDataHandler().updateInternalData(x_measured, True)

        current_torque = u_interp - 1.0 * mpc.Ks[0] @ model_handler.difference(
            x_measured, x_interp
        )

        simulator.reset()
        simulator.step(q, v, current_torque, dt_sim)

        q, v = simulator.state.qnew.copy(), simulator.state.vnew.copy()
        x_measured = np.concatenate((q, v))

    # Change current target in MPC
    target_new.translation[0] = target.translation[0] + x_mov * np.sin(
        np.pi * t * freq_mov / 180
    )
    target_new.translation[2] = target.translation[2] + z_mov * np.cos(
        np.pi * t * freq_mov / 180
    )

    vizer.visual_model.geometryObjects[
        target_id1
    ].placement.translation = target_new.translation

    start = time.time()
    mpc.iterate(x_measured)
    end = time.time()
    print(end - start)
    total += end - start
    vizer.display(q)

total = total / Tmpc
print("Mean time")
print(total)
