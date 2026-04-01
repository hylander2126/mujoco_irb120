# Set up GPU rendering.
# Configure MuJoCo to use the EGL rendering backend (requires GPU)
print('Setting environment variable to use GPU rendering:')

import mujoco

# Other imports and helper functions
import numpy as np

from mujoco_irb120.common.load_obj_in_env import load_environment, load_photoshoot
from mujoco_irb120.common.trajectory_recorder import TrajectoryRecorder
import mujoco_irb120.common.robot_controller as robot_controller
from mujoco_irb120.common.helper_fns import *
from mujoco_irb120.common.render_opts import RendererViewerOpts

# Graphics and plotting.
import mediapy as media
import matplotlib.pyplot as plt
import json as _json

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)
# Set matplotlib font size
fonts = {'size' : 20}
plt.rc('font', **fonts)
# %matplotlib notebook





# Enable/disable keyboard control toggle
KEYBOARD_CONTROL = True   # Set to False to disable keyboard control
print(f"Keyboard control: {'ENABLED' if KEYBOARD_CONTROL else 'DISABLED'}")
if KEYBOARD_CONTROL:
    print("  Controls: Arrow keys to move in X and Z directions")
    print("  Make sure viewer window has focus for keyboard input")





# ======================== Toggle visualization here =========================
VIZ = True   # set to False to record video without showing the viewer
MOTION_MODE = None   # Set to 'RECORD' to record keyboard motion, 'PLAYBACK' to replay from recorded_motion.npz
RECORD_FORCES = True   # Set to True to also record F/T sensor data during manual control
# ============================================================================

## Let's recall the model to reset the simulation
# 0: box_exp, 10: heart, 11: L_shape, 12: monitor, 13: soda, 14: flashlight
OBJECT = 0

model, data = load_environment(num=OBJECT, launch_viewer=False)

## =================== LOAD GROUND TRUTH PARAMS FROM JSON ===================
_obj_params = _json.load(open("object_params.json"))["objects"][str(OBJECT)]

com_gt   = np.subtract(_obj_params["com_gt_onshape"], _obj_params["com_gt_offset"])
m_gt     = _obj_params["mass_gt"]
init_xyz = np.array(_obj_params["init_xyz"])
## ===================================================================

## =================== SET FRICTION AT RUNTIME ===================
MU_TABLE = 0.2   # object-table sliding friction (override here to test different regimes)
model.geom_friction[model.geom("table").id, 0] = MU_TABLE

# Effective frictions for reference:
mu_obj        = model.geom_friction[model.ngeom - 1, 0]  # object geom sliding friction
mu_gt         = np.sqrt(MU_TABLE * mu_obj)               # effective object-table contact friction
mu_finger_obj = np.sqrt(model.geom_friction[model.geom("push_rod").id, 0] * mu_obj)  # effective finger-object
print(f"Table friction set to: {MU_TABLE}")
print(f"Effective object-table mu: {mu_gt:.3f}")
print(f"Effective finger-object mu: {mu_finger_obj:.3f}")
## ================================================================

## Setup based on robot model
irb = robot_controller.controller(model, data)

# Get initial EE pose (finger tip)
T_home = irb.FK()
print('Initial end-effector pose:\n', T_home)

## Set robot just in front of payload (same orientation as home position (facing +x))
T_init = T_home.copy()
T_init[:3, 3] = init_xyz.copy()

q_init = irb.IK(T_init, method=2, damping=0.5, max_iters=1000) # DLS method
irb.set_pose(q=q_init)

## The end pose we want to reach FOR POSITION CONTROL (format: 4x4 matrix)
T_end = T_init.copy()
T_end[0, 3] += 0.10  # Move EE forward by 15 cm in x direction

target_q = irb.IK(T_end, method=2, damping=0.5, max_iters=1000)  # DLS method

## TARE / Bias sensor
irb.ft_bias(n_samples=200)

## FOR VELOCITY CONTROL (format: [wx wy wz vx vy vz])
# target_vel  = np.array([0.0, 0.0, 0.0, 0.14, 0.0, 0.0])  # Move EE forward at 4 cm/s in x direction

## Initialize time, force and tilt history for plotting
t_hist          = []
w_hist          = []
quat_hist       = []
ball_pose_hist  = []  # (4,4) pose of ball-center site in world frame
sens_pose_hist  = []  # (4,4) pose of FT sensor site in world frame
con_bool_hist   = []  # contact flag
obj_pose_hist   = []  # (4,4) object pose in world frame (mj internal)

traj_duration = 6.0 # seconds
run_duration = traj_duration + 50.0 # 4.0  # seconds

# Initialize trajectory recorder for recording or load trajectory for playback
recorder = None
loaded_trajectory = None

if MOTION_MODE == 'RECORD':
    recorder = TrajectoryRecorder(irb)
    recorder.start_recording(verbose=False, record_forces=RECORD_FORCES)
elif MOTION_MODE == 'PLAYBACK':
    recorder = TrajectoryRecorder(irb)
    try:
        loaded_trajectory = recorder.load_trajectory("recorded_motion.npz")
        recorder.start_playback(trajectory=loaded_trajectory)
        print("Loaded and ready for playback.")
    except FileNotFoundError:
        print("ERROR: recorded_motion.npz not found. Skipping playback.")
        MOTION_MODE = None

## Additions for video recording
rv = RendererViewerOpts(model, data, vis=VIZ, show_left_UI=True)
# ===========================================================================
with rv: # enters viewer if vis=True, sets viewer opts, and readies offscreen renderer for video capture
    while rv.viewer_is_running() and not irb.stop and data.time < run_duration:
        irb.check_topple()                          # Check for payload topple condition

        # Apply control: either keyboard-based or trajectory-based position control
        if KEYBOARD_CONTROL and VIZ:
            v_cmd = rv.get_keyboard_input()
            irb.apply_cartesian_keyboard_ctrl(v_cmd, maintain_orientation=True, verbose=False)
        else:
            if data.time < traj_duration:
                alpha = data.time / traj_duration
                interp_q = (1 - alpha) * q_init + alpha * target_q
            else:
                interp_q = target_q.copy()
            irb.set_pos_ctrl(interp_q, check_ellipsoid=False)

        # Record or playback waypoint if enabled
        if MOTION_MODE == 'RECORD' and recorder:
            recorder.record_waypoint(record_type='joints')
        elif MOTION_MODE == 'PLAYBACK' and recorder:
            recorder.playback_step(playback_type='joints', interpolate=True)

        mujoco.mj_step(model, data)                 # Step the simulation

        w_hist.append(irb.ft_get_reading())
        quat_hist.append(irb.get_payload_pose(out='quat'))
        t_hist.append(data.time)
        ball_pose_hist.append(irb.get_site_pose("ball"))
        sens_pose_hist.append(irb.get_site_pose("sensor"))
        con_bool_hist.append(irb.check_contact())
        obj_pose_hist.append(irb.get_payload_pose(out='T'))

        rv.sync()
        rv.capture_frame_if_due(data)

t_hist          = np.asarray(t_hist,         dtype=float)
quat_hist       = np.asarray(quat_hist,      dtype=float)
con_bool_hist   = np.asarray(con_bool_hist,  dtype=float)
w_hist          = np.asarray(w_hist,         dtype=float).reshape(-1, 6)
ball_pose_hist  = np.asarray(ball_pose_hist, dtype=float).reshape(-1, 4, 4)
sens_pose_hist  = np.asarray(sens_pose_hist, dtype=float).reshape(-1, 4, 4)
obj_pose_hist   = np.asarray(obj_pose_hist,  dtype=float).reshape(-1, 4, 4)

ball_pos_hist = ball_pose_hist[:, :3, 3]
sens_pos_hist = sens_pose_hist[:, :3, 3]
obj_pos_hist  = obj_pose_hist[:,  :3, 3]

# Save trajectory if recording was enabled (skip if playback mode)
if MOTION_MODE == 'RECORD' and recorder:
    recorder.stop_recording(verbose=False)
    recorder.save_trajectory("recorded_motion.npz", format='numpy')
elif MOTION_MODE == 'PLAYBACK' and recorder:
    recorder.stop_playback()

print(f'\nSimulation ended in t = {data.time:.2f} seconds.')




# # ====== OPTIONAL: Save all simulation variables to numpy file ======
# # This captures all the data collected during the simulation loop
# # Useful when you want to preserve force/pose history even if not using trajectory recorder

np.savez(
    "simulation_data.npz",
    t_hist=t_hist,
    w_hist=w_hist,                    # Force/torque at each step (N, Nm)
    quat_hist=quat_hist,              # Object quaternion at each step
    ball_pose_hist=ball_pose_hist,    # Ball-center pose (4x4 transforms)
    sens_pose_hist=sens_pose_hist,    # FT sensor pose (4x4 transforms)
    con_bool_hist=con_bool_hist,      # Contact status at each step
    obj_pose_hist=obj_pose_hist,      # Object pose (4x4 transforms)
    ball_pos_hist=ball_pos_hist,      # Ball position trajectory
    sens_pos_hist=sens_pos_hist,      # Sensor position trajectory
    obj_pos_hist=obj_pos_hist         # Object position trajectory
)
print("Saved all simulation data to simulation_data.npz")


media.show_video(rv.frames, fps=rv.framerate)



