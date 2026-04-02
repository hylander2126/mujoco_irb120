# Prompt: Autonomous Multi-Phase Interaction Controller

## Context

I'm building a system where an ABB IRB120 robot arm uses a single fingertip (ball end-effector) to interact with unknown objects on a table in MuJoCo simulation. The goal is to collect force/torque and pose data across multiple interaction phases (push, squash-pull/tip) so that a downstream estimator can jointly fit mass, center-of-mass, and friction coefficient.

This is a **new branch** — the existing simulation loop in `simulation.ipynb` uses either keyboard teleop or a single straight-line position-control trajectory. We are replacing that with an **autonomous state-machine controller** that sequences through interaction phases without human input.

---

## Implementation Status

### Files Created

- **`src/mujoco_irb120/common/phase_controller.py`** — `PhaseController` class (main deliverable)
- **`scripts/simulation_multiphase.py`** — standalone runner script

### What Works (confirmed by user)

- **Robot moves** — differential-IK accumulator pattern (`_q_des`) works correctly.
- **APPROACH_PUSH** — 3-waypoint safe path (retract x → lower z → advance to standoff) avoids passing through the object.
- **PUSH phase** — robot moves in +x at push speed (`PUSH_SPEED_CTRL = 0.03 m/s`), detects contact via F/T magnitude, transitions after 4 cm of post-contact travel.
- **Pre-push dwell** — robot pauses `PRE_PUSH_DWELL = 1.0 s` at standoff before starting the push.
- **RETREAT** — 3-waypoint path clears the object and repositions above it. Re-observes object position after push via `_update_obj_geometry()` so post-push displacement is accounted for.
- **DESCEND** — robot descends slowly (`DESCEND_SPEED_CTRL = 0.01 m/s`) and stops on contact.
- **SQUASH** — proportional speed control (slows as force approaches target, `SQUASH_SPEED_MAX = 0.005 m/s`) reaches the force target without overshoot. `_q_des` resets on SQUASH entry to avoid descend momentum carrying over.
- **EE orientation maintenance** — `_move_toward_pos` captures `_R_des` at phase entry and applies a proportional angular velocity correction (`ORI_KP = 2.0`) each step to resist wrist drift.
- **Force frame correction** — `_get_ft_world()` rotates sensor readings into world frame so `ft_world[2]` is always the true vertical force (the sensor z-axis points in world +x, not world +z, at the nominal robot pose).
- **Phase settle window** — safety check is suppressed for 100 ms after each phase transition to avoid spurious force spikes from rapid deceleration.

### What Has NOT Been Tested / Confirmed

- **PULL_TIP phase** — pull-tip lateral motion with simultaneous z force maintenance. The z force sign logic was fixed (asymmetric rate: 5 mm/s down, 1 mm/s up) but has not been confirmed working end-to-end.
- **Slip detection** — sliding-window slip detector implemented but not exercised.
- **Slip retry loop** — squash force escalation on slip (×1.5, capped at `F_SQUASH_MAX = 12 N`) implemented but not tested.
- **DONE / summary** — `print_summary()` and `pc.save()` implemented but output not verified.
- **Other objects** — only tested with `OBJECT = 0` (box_exp). Other objects (heart, L-shape, flashlight) not tried.

---

## Existing Codebase You Must Understand First

Before writing any code, read and understand:

1. **`simulation.ipynb`** — The current sim loop. Key things to note:
   - `load_environment(num=OBJECT)` loads the MuJoCo model and data
   - `robot_controller.controller(model, data)` returns an `irb` controller object
   - `irb.FK()` → 4×4 end-effector pose
   - `irb.IK(T_target, method=2, damping=0.5, max_iters=1000)` → joint angles
   - `irb.set_pos_ctrl(q)` → position control
   - `irb.apply_cartesian_keyboard_ctrl(v_cmd, maintain_orientation=True)` → velocity control (used for keyboard; shows the velocity control interface exists)
   - `irb.ft_bias(n_samples=200)` → tare the F/T sensor
   - `irb.ft_get_reading()` → returns (6,) wrench [fx fy fz tx ty tz] **in sensor frame** (NOT world frame — see note below)
   - `irb.get_payload_pose(out='quat')` or `out='T'` → object pose
   - `irb.get_site_pose("ball")` → fingertip/ball pose (4×4)
   - `irb.get_site_pose("sensor")` → F/T sensor pose (4×4)
   - `irb.check_contact()` → boolean contact flag
   - `irb.check_topple()` → checks if object has toppled
   - The sim loop calls `mujoco.mj_step(model, data)` each iteration
   - Data is saved to `simulation_data.npz` with fields: `t_hist, w_hist, quat_hist, ball_pose_hist, sens_pose_hist, con_bool_hist, obj_pose_hist, ball_pos_hist, sens_pos_hist, obj_pos_hist, com_gt, mass_gt, mu_gt`
   - Object ground-truth params come from `object_params.json`

2. **`robot_controller.py`** — Explore the controller class to understand what methods are available (FK, IK, velocity control, force reading, contact checking, etc.). Do NOT modify this file unless absolutely necessary. If you need new low-level capabilities, ask me first.

3. **`com_estimation.py`** — The wrench models. You don't need to modify this, but understand `model_bkwd_wrench` and `model_fwd_wrench` to know what data the downstream estimator needs.

4. **`object_params.json`** — Contains per-object init_xyz (starting EE position), com_gt, mass_gt. The init_xyz is the position to place the EE just in front of the object before interaction begins.

5. **`load_obj_in_env.py`** — Understand how objects are loaded and what geometry info is available (bounding box, mesh extents, etc.).

---

## Critical Implementation Notes (Lessons Learned)

### Motion control pattern
`irb.set_pos_ctrl(q)` sets **absolute** joint targets for a PD controller. A single step of `q_curr + q_dot * dt` produces a target offset so small (~0.0001 rad) that the PD torque is negligible against gravity. **You must accumulate into a persistent `_q_des` across steps** (same as `apply_cartesian_keyboard_ctrl`'s `kb_q_des`), so the target drifts progressively away from the actual position and the PD controller sees growing error.

### F/T sensor frame
`irb.ft_get_reading()` returns forces in the **sensor's local frame**, not world frame. At the nominal robot pose, the sensor z-axis points in world +x (push direction) and sensor y-axis points in world +z (vertical). This means `ft[2]` is the horizontal push force and `ft[1]` is the vertical force — the opposite of what you'd expect. Always rotate to world frame before interpreting by axis:
```python
R_sensor = data.site_xmat[irb.ft_site].reshape(3, 3)
f_world  = R_sensor @ ft_sensor[:3]
# f_world[2] is now always vertical regardless of wrist orientation
```
This is implemented as `_get_ft_world()` in `phase_controller.py`.

### IK targets the EE flange, not the ball
`irb.IK()` minimises error at `site:tool0` (EE flange), not `site:ball_center`. The ball site is offset from the flange. To put the ball at a target position, compute `flange_target = ball_target + (T_ee[:3,3] - T_ball[:3,3])` first.

### Object geometry after push
`_scan_object_geometry()` reads the object AABB from MuJoCo's current state. This must be called again after the push phase (before planning the retreat/descend targets) because the object will have slid. This is done via `_update_obj_geometry()` at the start of RETREAT.

### Phase transition transients
Fast approach/retreat moves (up to 0.08 m/s) produce inertial F/T spikes of 30+ N at deceleration. The safety check must be suppressed for ~100 ms after each phase transition (`_phase_settle_until`).

### Squash momentum
When entering SQUASH from DESCEND, `_q_des` must be reset (not carried over), otherwise the downward velocity accumulated during descent causes force overshoot.

---

## What to Build (original spec, partially complete)

Create a new Python module: `mujoco_irb120/common/phase_controller.py`

This module implements a `PhaseController` class that autonomously sequences through interaction phases. It will be called from the simulation loop (which we'll also update in a new notebook or script).

### State Machine Phases

```
IDLE → SCAN → APPROACH_PUSH → PUSH → RETREAT_TO_TOP → DESCEND → SQUASH → PULL_TIP → DONE
```

#### Phase Details

**SCAN (optional/simple):**
- The robot is already positioned near the object (from `init_xyz` in object_params.json).
- For now, just record the object's initial pose and bounding box info from the MuJoCo model. We can add vision later.
- Determine: object centroid height (from geom), object top surface height, object front face position.
- Transition: immediately to APPROACH_PUSH.

**APPROACH_PUSH:**
- Move the EE to a position offset from the object's front face, at approximately the centroid height of the object (half the object's height from the table).
- Use position control (IK → `set_pos_ctrl`) to move there.
- Transition: when EE is within tolerance of the target pose.

**PUSH:**
- Command constant forward velocity (in +x direction) using position control with small incremental targets, or velocity control if available.
- Push speed: ~0.02–0.05 m/s (quasi-static).
- Monitor F/T sensor for contact onset (force magnitude exceeds a noise threshold, e.g., 0.5 N).
- Continue pushing for a fixed distance after contact (e.g., 3–5 cm) or until the object begins to tip (detected by pitch angle change > some threshold like 2°) or slide (detected by F/T plateau — force stops increasing despite continued displacement).
- **Regardless of whether the object slides or tips, record the data and move to next phase.**
- Transition: push distance reached, or tip detected, or slide detected, or force exceeds safety limit (e.g., 20 N).

**RETREAT_TO_TOP:**
- Retract the EE back (in -x) to clear the object, then move up (+z) to above the object's top surface.
- Then move forward (+x) to be directly above the object's top surface, near the edge closest to the robot.
- Use position control.
- Transition: when EE is at target position above object top.

**DESCEND:**
- Move EE downward (-z) slowly until contact with the object top surface is detected (F/T z-force exceeds threshold).
- Transition: contact detected.

**SQUASH:**
- Continue pressing down to achieve a target normal force `F_squash_target` (start with 2–5 N).
- Use simple proportional force control: if current F_z < target, move down a small increment; if F_z > target, hold position.
- Hold for a short duration (0.5 s) to get steady-state reading.
- Transition: force target achieved and held stable for the specified duration.

**PULL_TIP:**
- While maintaining downward force (squash), command lateral motion (in +x, away from robot, to pull the top of the object and induce tipping toward the robot).
- Pull speed: ~0.01–0.03 m/s.
- Monitor for:
  - **Successful tipping**: object pitch angle changes significantly (> 5°). Continue pulling to collect tipping trajectory data up to ~30-45° or until the robot can't maintain contact.
  - **Finger slip detection**: the EE moves laterally but the object's orientation (pitch angle) does NOT change AND force drops suddenly. This means the finger slipped off.
- Transition on success: tipping angle reaches target → DONE.
- **Transition on slip: reset.** Go back to RETREAT_TO_TOP, but multiply `F_squash_target` by 1.5. Repeat the squash-pull cycle. Cap retries at 3-4 attempts.

**DONE:**
- Stop all motion, record final data.

### Data Recording

Throughout ALL phases, record at every sim step (same as current simulation loop):
- `t_hist`, `w_hist`, `quat_hist`, `ball_pose_hist`, `sens_pose_hist`, `con_bool_hist`, `obj_pose_hist`

Additionally, record a **phase label** for each timestep:
- `phase_hist`: array of integers or strings labeling which phase each sample belongs to (e.g., 0=IDLE, 1=SCAN, 2=APPROACH_PUSH, 3=PUSH, 4=RETREAT, 5=DESCEND, 6=SQUASH, 7=PULL_TIP, 8=DONE).

This phase label is critical for the downstream segmentation — we need to know which data came from which interaction.

Save everything to `simulation_data_multiphase.npz` with the same fields as before PLUS `phase_hist`.

### Key Implementation Details

1. **Quasi-static assumption**: all motions must be slow. The robot should never slam into the object. Use velocity limits and ramp up/down.

2. **Contact detection**: use `irb.check_contact()` AND F/T magnitude. The F/T sensor is more reliable for detecting contact onset; the boolean flag may have a delay.

3. **Slip detection heuristic**: during PULL_TIP, compute a running window (last ~0.5 s) of:
   - EE lateral displacement: `delta_x_ee = |x_ee(t) - x_ee(t-window)|`
   - Object pitch change: `delta_pitch = |pitch(t) - pitch(t-window)|`
   - If `delta_x_ee > some_threshold` (e.g., 2 mm) AND `delta_pitch < some_threshold` (e.g., 0.5°), declare slip.

4. **Force control**: we don't have a built-in force controller. Implement a simple proportional scheme:
   ```python
   # In squash phase, to maintain target F_z:
   f_z_current = abs(ft_world[2])  # world-frame z-component
   error = F_squash_target - f_z_current
   # proportional speed (slows as force approaches target)
   speed = SQUASH_SPEED_MAX * (error / F_squash_target)
   ```
   Use `_get_ft_world()` — never use raw `ft_get_reading()[2]` for vertical force.

5. **Object geometry**: get the object's bounding box from MuJoCo geom data to determine centroid height, top surface, front face, etc. Re-read after push to get updated position.

6. **Safety**: add force limits. If any force component exceeds 30 N, abort the current phase and retreat. If the object topples completely (pitch > 60°), stop. Suppress the safety check for 100 ms after each phase transition.

### Integration

Create a new script or notebook `simulation_multiphase.py` (or `.ipynb`) that:
1. Loads the environment (same as current `simulation.ipynb`)
2. Instantiates the `PhaseController`
3. Runs the sim loop, calling `phase_controller.step()` each iteration instead of the keyboard/trajectory control
4. The `step()` method returns the control action (joint positions or velocity command) based on current phase and sensor readings
5. After the loop, saves the multiphase data

The sim loop structure should look roughly like:

```python
pc = PhaseController(irb, model, data, object_id=OBJECT)

while running and not pc.is_done():
    action = pc.step()  # internally reads sensors, updates state machine, returns control
    mujoco.mj_step(model, data)
    pc.record()  # appends current timestep data
    # viewer sync, etc.

pc.save("simulation_data_multiphase.npz")
```

### What NOT to Do

- Do NOT modify `robot_controller.py` unless you absolutely must (and explain why).
- Do NOT modify `com_estimation.py` — that's the downstream estimator, separate concern.
- Do NOT try to make the controller "smart" or use ML. This is a simple heuristic state machine. Dumb and reliable.
- Do NOT worry about optimizing the interaction sequence or choosing which face to push. Just push in +x for now.
- Do NOT add perception/vision — we'll handle that later.

### Testing

After implementation, test with `OBJECT = 0` (box_exp). The box is the simplest geometry. Run the full state machine and verify:
1. The robot pushes the object at centroid height
2. The robot retreats and repositions above the object
3. The robot squashes down and attempts to pull-tip
4. If slip occurs, it retries with higher squash force
5. Data is saved with phase labels

Print a summary at the end: which phases were executed, duration of each, whether tip was achieved, how many slip retries occurred.
