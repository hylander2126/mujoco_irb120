"""
simulation_multiphase.py
------------------------
Autonomous multi-phase data collection for mass / CoM / friction estimation.

Run from the repo root or scripts/ directory:

    cd scripts/phase-state-machine/
    python simulation_multiphase.py

Outputs:
  - simulation_data_multiphase.npz          (sensor + pose data + phase labels)
  - phase_controller_<timestamp>.log        (human-readable state machine log)
  - simulation_<timestamp>.mp4  (optional)  (video, if RECORD_VIDEO=True)
"""

import mujoco
import numpy as np
import json as _json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mujoco_irb120.util.load_obj_in_env import load_environment
import mujoco_irb120.controllers.robot_controller as robot_controller
from mujoco_irb120.controllers.phase_controller import PhaseController, Phase
from mujoco_irb120.util.render_opts import RendererViewerOpts

np.set_printoptions(precision=3, suppress=True, linewidth=100)


# ===========================================================================
# Configuration
# ===========================================================================

OBJECT          = 10        # 0=box_exp, 10=heart, 11=L_shape, 14=flashlight
VIZ             = 1     # Open the MuJoCo viewer (set False for headless / faster runs)
RECORD_VIDEO    = not VIZ  # Save an mp4 of the offscreen render (requires: pip install mediapy)
MU_TABLE        = 0.2      # Sliding friction coefficient for table geom
MAX_SIM_TIME    = 120.0    # Hard sim-time timeout (seconds)
OUTPUT_FILE     = "simulation_data_multiphase.npz"

# Set to a Phase to skip earlier phases and start there directly.
# The robot must already be in a sensible pose for the chosen phase.
# Options: None (full run), Phase.RETREAT, Phase.DESCEND, Phase.SQUASH, Phase.PULL_TIP
START_PHASE     = None #Phase.RETREAT


# ===========================================================================
# Setup
# ===========================================================================

print(f"Loading environment for object {OBJECT}...")
model, data = load_environment(num=OBJECT, launch_viewer=False)
assert model is not None, "Failed to load environment."

# model.geom_friction[model.geom("table").id, 0] = MU_TABLE

irb = robot_controller.controller(model, data)

_params  = _json.load(open(_REPO_ROOT / "src" / "mujoco_irb120" / "assets" / "object_params.json"))["objects"][str(OBJECT)]
init_xyz = np.array(_params["init_xyz"])

# Place robot at init pose and tare sensor
T_home = irb.FK()
T_init = T_home.copy()
T_init[:3, 3] = init_xyz
q_init = irb.IK(T_init, method=2, damping=0.5, max_iters=1000)
irb.set_pose(q=q_init)
irb.ft_bias(n_samples=200)

pc = PhaseController(irb, model, data, object_id=OBJECT)

# --- Log file ---
_log_dir  = Path(__file__).parent
_log_name = f"phase_controller.log"
_log_path = str(_log_dir / _log_name)
_log_path = pc.set_log_file(_log_path)
print(f"Logging to: {_log_path}")

# --- Start phase ---
if START_PHASE is not None:
    print(f"\nSkipping to {START_PHASE.name} — robot must already be in position.")
    pc.start_at_phase(START_PHASE)
    start_label = START_PHASE.name
else:
    start_label = "IDLE (full run)"

print(f"\nStarting from: {start_label}  (max {MAX_SIM_TIME} s sim time)")
print("Phases: IDLE→SCAN→APPROACH_PUSH→PUSH→RETREAT→DESCEND→SQUASH→PULL_TIP→DONE\n")


# ===========================================================================
# Simulation loop
# ===========================================================================

with RendererViewerOpts(model, data, vis=VIZ, show_left_UI=False) as rv:
    while rv.viewer_is_running() and not pc.is_done() and data.time < MAX_SIM_TIME:
        pc.step()
        mujoco.mj_step(model, data)
        pc.record()
        rv.sync()
        if RECORD_VIDEO:
            rv.capture_frame_if_due(data)


# ===========================================================================
# Post-run
# ===========================================================================

print(f"\nSimulation ended at t = {data.time:.2f} s.")
pc.print_summary()

_out_path = str(_log_dir / OUTPUT_FILE)
pc.save(_out_path)
print(f"Log written to: {_log_path}")

if RECORD_VIDEO:
    _vid_name = f"simulation_video.mp4"
    rv.save_video(str(_log_dir / _vid_name))

print("Done.")
