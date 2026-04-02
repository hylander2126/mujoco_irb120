# CLAUDE.md — Project Context for claude-code

This file records key implementation decisions, lessons learned, and conventions
for the `mujoco_irb120` project so they are available in future sessions.

---

## Project Overview

ABB IRB120 robot arm in MuJoCo simulation. The primary goal is collecting F/T
and pose data through autonomous multi-phase interactions with tabletop objects
for downstream **mass / CoM / friction estimation**.

---

## Autonomous Phase Controller

**File:** `src/mujoco_irb120/common/phase_controller.py`
**Runner:** `scripts/phase-state-machine/simulation_multiphase.py`

### State machine sequence
```
IDLE → SCAN → APPROACH_PUSH → PUSH → RETREAT → DESCEND → SQUASH → PULL_TIP → DONE
```

### Critical implementation notes

#### Robot motion (most important)
The PD controller requires a **persistent joint target accumulator** (`_q_des`).
Calling `set_pos_ctrl(q_curr + q_dot*dt)` produces ~0.0001 rad steps → torque
too small → robot doesn't move. `_q_des` must be accumulated across steps, never
reset to `None` at the start of each call. It is only reset on phase transitions
(except PUSH and PULL_TIP which inherit it for continuity).

#### Sensor frame
`ft_get_reading()` returns forces in the **sensor's local frame**, not world frame.
Always rotate via `R_sensor @ f_S` (see `_get_ft_world()`).  
At nominal pose: sensor-y ≈ world-z (vertical), sensor-z ≈ world-x (push direction).  
`ft_world[2]` is the vertical (squash) force component.

#### Jacobian ordering
`irb.get_jacobian()` returns `[J_rot; J_pos]`, so `v_cmd = [wx wy wz vx vy vz]`.

#### Orientation maintenance
Capture `_R_des` at phase entry; each step compute
`w_ori = ORI_KP * R.from_matrix(R_des @ R_curr.T).as_rotvec()` and include in
`v_cmd[:3]`. Without this the EE tilts under joint compliance during squash.

#### Phase settle window
Suppress safety force check for 100 ms after every phase transition
(`_phase_settle_until`) to avoid triggering on inertial transients from fast moves.
DESCEND additionally ignores contact for an extra 100 ms (`settle_until + 0.1 s`).

#### Squash → PULL_TIP continuity
`_q_des` must **not** be reset on PULL_TIP entry — it carries the squash depth.
`_q_squash` captures joint state at squash completion as a z-floor.
PULL_TIP uses lateral-only `v_cmd` (vx only, no vz) to avoid kinematic coupling
that would otherwise lift the EE. A 300 ms stabilisation hold is inserted before
lateral motion begins.

#### Squash position (x)
The robot squashes near the **front face** of the object (the same face it pushed),
not the centroid. `obj_squash_x = obj_front_x + ball_radius`. This maximises the
tipping moment arm for PULL_TIP.

#### Slip detection
- Gated on `SLIP_MIN_TRAVEL = 0.010 m` minimum lateral travel before any slip
  can be declared (avoids false positives during stabilisation).
- Disabled entirely once `cumulative_pitch > 0.1°` from pull start — if the object
  is already rotating, that is tipping, not slip.
- On slip: do **not** retract in x. Lift straight up above the object
  (`_slip_lift = True` flag in RETREAT), then descend again.

#### Object geometry
`_scan_object_geometry()` reads MuJoCo AABB from geom data. Sets:
- `obj_top_z`, `obj_centroid_z`, `obj_front_x`, `obj_center_x`, `obj_half_x`
- `obj_squash_x = obj_front_x + ball_radius`

Called again at RETREAT entry via `_update_obj_geometry()` to account for
object displacement after push.

### Tunable constants (current values)
```python
MOVE_SPEED          = 0.08    # m/s — fast approach/retreat
PUSH_SPEED_CTRL     = 0.03    # m/s — push speed
DESCEND_SPEED_CTRL  = 0.01    # m/s — slow descent to avoid overshoot
SQUASH_SPEED_MAX    = 0.005   # m/s — proportional squash speed cap
PULL_SPEED          = 0.02    # m/s — lateral pull speed
ORI_KP              = 2.0     # rad/s per rad — orientation correction gain
PRE_PUSH_DWELL      = 1.0     # s — pause at standoff before pushing
PREPUSH_GAP         = 0.005   # m — gap to front face (no pre-load at standoff)
SAFETY_FORCE_LIMIT  = 30.0    # N
F_SQUASH_INIT       = 3.0     # N — initial squash force target
SLIP_MIN_TRAVEL     = 0.010   # m
```

### Config flags in simulation_multiphase.py
```python
OBJECT       = 0           # 0=box_exp, 10=heart, 11=L_shape, 14=flashlight
VIZ          = True        # Open the MuJoCo passive viewer
RECORD_VIDEO = True        # Save mp4 via mediapy (pip install mediapy)
START_PHASE  = Phase.RETREAT  # None for full run; or Phase.RETREAT / DESCEND / SQUASH / PULL_TIP
MAX_SIM_TIME = 120.0       # Hard timeout (s)
```

### Log file
`pc.set_log_file(path)` — all `_log()` calls go to both stdout and the file.
A timestamped `.log` is created automatically in the script directory.

### Video recording
`rv.capture_frame_if_due(data)` fills `rv.frames` each step.  
`rv.save_video(path)` writes an mp4 with mediapy after the loop.  
Set `VIZ=False, RECORD_VIDEO=True` for headless fast runs with a saved video.

---

## Camera setup

Camera parameters are in `src/mujoco_irb120/common/render_opts.py`:
```python
CAM_DISTANCE  = 1.5
CAM_ELEVATION = -30
CAM_AZIMUTH   = 90
CAM_LOOKAT    = np.array([0.75, 0, 0.25])
```

**To read the current view from the live viewer**, add inside the sim loop:
```python
print(f"distance={rv.viewer.cam.distance:.3f}  elevation={rv.viewer.cam.elevation:.1f}  "
      f"azimuth={rv.viewer.cam.azimuth:.1f}  lookat={rv.viewer.cam.lookat}")
```

**To pin a camera in the XML** (`src/mujoco_irb120/assets/table_push.xml`):
```xml
<camera name="default_view" pos="0.3 -1.2 0.7" xyaxes="1 0 0 0 0.5 1" fovy="45"/>
```
Then in `_apply_offscreen_opts`:
```python
cam_obj.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam_obj.fixedcamid = model.camera("default_view").id
```

---

## Key files

| File | Purpose |
|------|---------|
| `src/mujoco_irb120/common/phase_controller.py` | Autonomous state machine |
| `src/mujoco_irb120/common/render_opts.py` | Viewer + offscreen renderer, video saving |
| `src/mujoco_irb120/common/robot_controller.py` | IK, FK, Jacobian, F/T, set_pos_ctrl |
| `scripts/phase-state-machine/simulation_multiphase.py` | Runner script |
| `scripts/object_params.json` | Per-object ground-truth mass, CoM, init pose |
| `src/mujoco_irb120/assets/table_push.xml` | Main scene template |
| `src/mujoco_irb120/assets/my_objects/box/box_exp.xml` | Box: pos="0.62 0 0.2", half-extents 0.05×0.05×0.15 |

---

## Status (as of last session)

| Feature | Status |
|---------|--------|
| APPROACH_PUSH → PUSH | Working |
| RETREAT waypoints | Working |
| DESCEND contact detection | Working |
| SQUASH force control + orientation hold | Working |
| SQUASH → PULL_TIP continuity (no force drop) | Implemented, needs end-to-end verification |
| Slip detection with min-travel + cumulative-pitch gates | Implemented |
| Slip recovery: lift straight up, no x-retract | Implemented |
| Squash at front edge (max leverage) | Implemented |
| `start_at_phase()` for intermediate starts | Working |
| Log file output | Working |
| Video recording (`RECORD_VIDEO` flag) | Implemented |
| `VIZ=False` headless mode | Working (existing) |
