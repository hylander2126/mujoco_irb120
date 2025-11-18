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
# The FT has length 0.03m and the pusher has length 0.1143m
FINGER_OFFSET = 0.03 + 0.1143 # compensate
# ==================================================

COMPLETE_OFFSET = [-FINGER_OFFSET, 0, -Z_OFFFSET]
# On the real robot, the same home position will read Z_sim=Z_real but actually Z_real is
# HIGHER in physical space by Z_OFFSET. So Z_real = Z_sim - Z_OFFSET
# Similarly, on real robot, same home position will read X_real = X_sim - FINGER_OFFSET

# THEREFORE, to reach a desired real-world physical position we must:
# 1) Take our desired position in real world coordinates
# 2) SUBTRACT Z_OFFSET from Z_desired
# 3) SUBTRACT FINGER_OFFSET from X_desired

# e.g. real home position is [0.373, 0.000, 0.626]


# Let's see the initial pose (should be home)
temp = irb.FK()
temp[0:3,3] += COMPLETE_OFFSET
print("Initial pose (in flange frame):\n", temp)

DESIRED_XYZ = [0.8 , 0.0, 0.25] # DESIRED FINGERTIP POSITION (IGNORING OFFSETS)

desired_pose = np.eye(4)  # Example desired pose (identity matrix to keep home orientation)
desired_pose[0:3, 3] = [0.8 + FINGER_OFFSET, 0, 0.25 - Z_OFFFSET]  # Set desired position

desired_q = irb.IK(desired_pose, method=2, damping=0.5, max_iters=1000)

if desired_q is None:
    print("IK solution not found\n")
    exit(0)



print("Desired joint angles from IK:\n", np.round(desired_q, 3))
irb.set_pose(desired_q)

# Let's see what the robot looks like in this pose
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        irb.set_pose(desired_q)
        viewer.sync()