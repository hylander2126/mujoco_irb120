import utils.robot_controller as robot_controller
from load_obj_in_env import load_environment
import numpy as np
import mujoco
import mujoco.viewer


model, data = load_environment(num=0)

irb = robot_controller.controller(model, data)

# ==================================================
# NOTE: THIS IK USES A Z OFFSET TO COMPENSATE FOR THE TABLE HEIGHT
Z_OFFFSET = 0.035 # Robot mounting offset
# ==================================================

# ==================================================
# NOTE: THIS IK USES THE FINGERTIP POSITION AS THE FRAME OF REFERENCE
# The FT has length 0.08225 and the pusher has length 0.110
FINGER_OFFSET = 0.08225 + 0.110 # approx .19225 m total
# ==================================================

COMPLETE_OFFSET = [-FINGER_OFFSET, 0, -Z_OFFFSET]
# On the real robot, the same home position will read Z_sim=Z_real but actually Z_real is
# HIGHER in physical space by Z_OFFSET. So Z_real = Z_sim - Z_OFFSET
# Similarly, on real robot, same home position will read X_real = X_sim - FINGER_OFFSET

# THEREFORE, to reach a desired real-world physical position we must:
# 1) Take our desired position in real world coordinates
# 2) SUBTRACT Z_OFFSET from Z_desired
# 3) SUBTRACT FINGER_OFFSET from X_desired

# e.g. real home position is [0.374, 0.000, 0.630]


# Let's see the initial pose (should be home)
temp = irb.FK()
OG_POSE = temp.copy()
temp[2,3] -= Z_OFFFSET # TOOL FLANGE IN *ROBOT* FRAME (must compensate for mounting 
                                    # ROBOT IS NOW INSIDE THE TABLE AND ABOVE Z_W0 BY Z_OFFSET)
print("Initial pos (FLANGE):\n", temp[0:3, 3])

temp[0,3] += FINGER_OFFSET
# print("Initial pose (FINGERTIP):\n", temp)

DESIRED_XYZ = OG_POSE[0:3, 3].copy()
DESIRED_XYZ[0] -= 0.05
DESIRED_XYZ[2] = 0.175 #- Z_OFFFSET # DESIRED FINGERTIP POSITION (IGNORING OFFSETS)
desired_pose = np.eye(4)  # Example desired pose (identity matrix to keep home orientation)
desired_pose[0:3, 3] = DESIRED_XYZ
print("Desired pose (FLANGE):\n", desired_pose)

# TEMPORARY TESTING REAL ROBOT IK AND FK:
# known_q = np.zeros(6)
# irb.set_pose(known_q)
test_pose = irb.FK()
test_q = irb.IK(test_pose, method=2, damping=0.5, max_iters=1000)

# print("Known joint angles:\n", known_q)
print("IK solution for known pose:\n", test_q)
# print("Difference:\n", test_q - known_q)



desired_q = irb.IK(desired_pose, method=2, damping=0.5, max_iters=1000)

if desired_q is None:
    print("IK solution not found\n")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            irb.set_pose(np.zeros((6,1)))
            viewer.sync()
    exit(0)

irb.set_pose(desired_q)
print("Desired joint angles from IK:\n", np.round(desired_q, 3).flatten())
f_pos = irb.FK()[0:3, 3]
f_pos[0] += FINGER_OFFSET
print("Current Fingertip Position (from table/robot frame):\n", np.round(f_pos, 4))

# Let's see what the robot looks like in this pose
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE # Visualize BODY frames only
    model.vis.scale.framewidth      = 0.025  # Frame axis width
    model.vis.scale.framelength     = 0.75   # Frame axis length

    while viewer.is_running():
        irb.set_pos_ctrl(desired_q)
        # mujoco.mj_step(model, data)
        viewer.sync()