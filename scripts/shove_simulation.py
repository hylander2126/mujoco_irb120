# Set up GPU rendering.
# Configure MuJoCo to use the EGL rendering backend (requires GPU)
import argparse
import os
import sys
from pathlib import Path

_DEFAULT_MUJOCO_GL = "glfw" if "--show-viewer" in sys.argv else "egl"
os.environ.setdefault("MUJOCO_GL", _DEFAULT_MUJOCO_GL)
print(f"MuJoCo GL backend: {os.environ['MUJOCO_GL']}")

import mujoco

# Other imports and helper functions
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.scene import load_environment, load_photoshoot
from robot.controllers import robot as robot_controller
from util.helper_fns import *
from util.render_opts import RendererViewerOpts

# Graphics and plotting.
import mediapy as media
import matplotlib.pyplot as plt
import json as _json

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)
# Set matplotlib font size
fonts = {'size' : 20}
plt.rc('font', **fonts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the IRB120 shove simulation and record rollout/video frames."
    )
    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument(
        "--show-viewer",
        dest="show_viewer",
        action="store_true",
        help="Open the live MuJoCo viewer while the simulation runs.",
    )
    viewer_group.add_argument(
        "--no-viewer",
        dest="show_viewer",
        action="store_false",
        help="Run headless/offscreen and save the recorded video after the run.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Path for the rendered video file. Defaults to outputs/rollouts/shove_simulation.mp4.",
    )
    parser.add_argument(
        "--show-video",
        action="store_true",
        help="Also create an inline mediapy display object after saving the video.",
    )
    parser.set_defaults(show_viewer=False)
    return parser.parse_args()


ARGS = parse_args()


# ======================== Toggle visualization here =========================
VIZ = ARGS.show_viewer
# ============================================================================
print(f"Live viewer: {'ENABLED' if VIZ else 'DISABLED'}")

## Let's recall the model to reset the simulation
# 0: box_exp, 10: heart, 11: L_shape, 12: monitor, 13: soda, 14: flashlight
OBJECT = 0
ROLLOUT_DIR = REPO_ROOT / "outputs" / "rollouts"
OBJECT_PARAMS_PATH = REPO_ROOT / "environment" / "object_params.json"
SHOVE_DATA_PATH = ROLLOUT_DIR / "shove_simulation_data.npz"
SHOVE_VIDEO_PATH = ARGS.video_path or (ROLLOUT_DIR / "shove_simulation.mp4")

model, data = load_environment(num=OBJECT, launch_viewer=False)

## =================== LOAD GROUND TRUTH PARAMS FROM JSON ===================
_obj_params = _json.load(open(OBJECT_PARAMS_PATH))["objects"][str(OBJECT)]

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

## =================== SHOVE TEST PARAMETERS ===================
# Table top is at z=0.05 and the box spans z=0.05-0.35 (see robot/assets/objects/box/box_exp.xml),
# so the finger height must sit inside that range to make a clean planar push (not dig into the table).
FINGER_HEIGHT   = 0.18   # hardcoded finger (fingertip) height above table, in meters
SHOVE_VELOCITY  = 0.30   # m/s, adjustable shove speed in +x direction
SHOVE_DURATION  = 0.5    # seconds to hold the shove velocity before stopping
## ===============================================================

## Set robot just in front of payload (same orientation as home position (facing +x))
## at a fixed, hardcoded finger height rather than the payload's resting height.
T_init = T_home.copy()
T_init[:3, 3] = init_xyz.copy()
T_init[2, 3] = FINGER_HEIGHT

q_init = irb.IK(T_init, method=2, damping=0.5, max_iters=1000) # DLS method
irb.set_pose(q=q_init)

## TARE / Bias sensor
irb.ft_bias(n_samples=200)

## FOR VELOCITY CONTROL (format: [wx wy wz vx vy vz])
## NOTE: the robot's actuators are position-controlled, so the Cartesian velocity command
## must be integrated into a joint-position target (apply_cartesian_keyboard_ctrl does this)
## rather than sent directly via set_vel_ctrl, which would feed raw velocities into the
## position actuators as if they were joint-angle targets.

# shove_vel = np.array([0.0, 0.0, 0.0, SHOVE_VELOCITY, 0.0, 0.0])

shove_vel = np.zeros(6)

## Initialize time, force and tilt history for plotting
t_hist          = []
w_hist          = []
quat_hist       = []
ball_pose_hist  = []  # (4,4) pose of ball-center site in world frame
sens_pose_hist  = []  # (4,4) pose of FT sensor site in world frame
con_bool_hist   = []  # contact flag
obj_pose_hist   = []  # (4,4) object pose in world frame (mj internal)

run_duration = SHOVE_DURATION + 4.0  # seconds, includes settling time after the shove

## Additions for video recording
rv = RendererViewerOpts(model, data, vis=VIZ, show_left_UI=True)
# ===========================================================================
with rv: # enters viewer if vis=True, sets viewer opts, and readies offscreen renderer for video capture
    while rv.viewer_is_running() and not irb.stop and data.time < run_duration:
        irb.check_topple()                          # Check for payload topple condition

        # Shove at constant velocity for SHOVE_DURATION, then hold the last commanded pose.
        if data.time < SHOVE_DURATION:
            irb.apply_cartesian_keyboard_ctrl(shove_vel, maintain_orientation=True, verbose=False)
        else:
            irb.apply_cartesian_keyboard_ctrl(np.zeros(6), maintain_orientation=True, verbose=False)

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

print(f'\nSimulation ended in t = {data.time:.2f} seconds.')

ROLLOUT_DIR.mkdir(parents=True, exist_ok=True)
np.savez(
    SHOVE_DATA_PATH,
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
print(f"Saved all simulation data to {SHOVE_DATA_PATH}")

if not rv.frames:
    raise RuntimeError("No video frames were captured; cannot write shove simulation video.")

SHOVE_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
media.write_video(SHOVE_VIDEO_PATH, rv.frames, fps=rv.framerate)
print(f"Saved shove simulation video to {SHOVE_VIDEO_PATH}")

if ARGS.show_video:
    media.show_video(rv.frames, fps=rv.framerate)
