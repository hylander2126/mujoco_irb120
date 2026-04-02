"""
phase_controller.py
-------------------
Autonomous multi-phase interaction controller for the ABB IRB120 robot in MuJoCo.

State machine:
    IDLE → SCAN → APPROACH_PUSH → PUSH → RETREAT_TO_TOP → DESCEND → SQUASH → PULL_TIP → DONE

Usage (from sim loop):
    pc = PhaseController(irb, model, data, object_id=0)
    while not pc.is_done():
        pc.step()
        mujoco.mj_step(model, data)
        pc.record()
    pc.save("simulation_data_multiphase.npz")
"""

import json
import time
from collections import deque
from enum import IntEnum
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Phase enumeration
# ---------------------------------------------------------------------------

class Phase(IntEnum):
    IDLE          = 0
    SCAN          = 1
    APPROACH_PUSH = 2
    PUSH          = 3
    RETREAT       = 4
    DESCEND       = 5
    SQUASH        = 6
    PULL_TIP      = 7
    DONE          = 8


PHASE_NAMES = {p: p.name for p in Phase}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT  = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_PARAMS_FILE = _SCRIPTS_DIR / "object_params.json"


# ---------------------------------------------------------------------------
# PhaseController
# ---------------------------------------------------------------------------

class PhaseController:
    """
    Autonomous state-machine controller that sequences through multiple
    interaction phases (push, squash, pull-tip) to collect data for
    joint mass/CoM/friction estimation.

    Args:
        irb         : robot_controller.controller instance
        model       : mujoco.MjModel
        data        : mujoco.MjData
        object_id   : integer key into object_params.json (default 0 = box)
    """

    # ------------------------------------------------------------------
    # Tunable constants
    # ------------------------------------------------------------------
    # Approach / push
    PUSH_SPEED          = 0.03      # m/s forward push speed
    PUSH_DIST_AFTER_CONTACT = 0.04  # how far to push after contact (m)
    PUSH_FORCE_LIMIT    = 20.0      # N — abort push if exceeded
    SAFETY_FORCE_LIMIT  = 30.0      # N — emergency retreat
    TIP_ANGLE_ABORT     = 60.0      # deg — stop if object tips this far
    TIP_DETECT_DEG      = 2.0       # deg — tip detected during push
    CONTACT_FORCE_THRESH= 0.5       # N — force magnitude for contact onset
    APPROACH_TOL        = 0.005     # m — position tolerance to advance phase
    PREPUSH_GAP        = 0.005     # m — initial gap to object before pushing (no pre-load)
    PRE_PUSH_DWELL      = 1.0       # s — pause at standoff before starting push

    # Retreat / top
    RETREAT_CLEARANCE   = 0.06      # m — pull back from object in -x before going up
    TOP_CLEARANCE       = 0.03      # m — how far above object top to hover

    # Descend
    DESCEND_SPEED       = 0.02      # 0.01 m/s
    DESCEND_CONTACT_F   = 0.5       # N — contact force for descent stop

    # Squash / force control
    F_SQUASH_INIT       = 3.0       # N initial squash force target
    F_SQUASH_MAX        = 12.0      # N cap on retried squash force
    F_SQUASH_KP         = 0.0001    # m/N proportional gain
    SQUASH_HOLD_TIME    = 0.5       # s — hold at target before transitioning

    # Pull-tip
    PULL_SPEED          = 0.02      # m/s lateral pull speed
    TIP_SUCCESS_DEG     = 5.0       # deg pitch to declare tipping started
    TIP_DONE_DEG        = 35.0      # deg pitch to stop pulling
    SLIP_WINDOW         = 0.5       # s sliding window for slip detection
    SLIP_EE_THRESH      = 0.002     # m EE lateral movement within window to flag slip
    SLIP_PITCH_THRESH   = 0.5       # deg pitch change below which slip declared
    SLIP_MIN_TRAVEL     = 0.010     # m minimum total lateral travel before slip can be declared
    MAX_SLIP_RETRIES    = 3

    # Speed limits for quasi-static motion
    MOVE_SPEED          = 0.08      # m/s — max Cartesian speed for approach / retreat moves
    PUSH_SPEED_CTRL     = 0.03      # m/s — push speed
    DESCEND_SPEED_CTRL  = 0.01      # m/s — descend speed (slow to avoid overshoot on contact)
    SQUASH_SPEED_MAX    = 0.005     # m/s — max speed during squash force control
    ORI_KP              = 2.0       # rad/s per rad of orientation error — restores EE orientation

    def __init__(self, irb, model: mujoco.MjModel, data: mujoco.MjData, object_id: int = 0):
        self.irb    = irb
        self.model  = model
        self.data   = data
        self.object_id = object_id

        # Load ground-truth params
        self._load_params(object_id)

        # Current phase
        self.phase = Phase.IDLE

        # Logging
        self._log_file = None
        self._log_path = None

        # Phase timing
        self._phase_start_time: dict[Phase, float] = {}
        self._phase_end_time:   dict[Phase, float] = {}
        self._phase_settle_until: float = 0.0   # ignore safety check until this sim time
        self._pull_stable_until: float = 0.0   # PULL_TIP: hold z before lateral motion
        self._pull_start_x: float = None       # ball x at start of lateral pull (for min-travel gate)
        self._q_squash: np.ndarray = None      # joint target captured at squash completion (z floor)

        # Geometry info (filled during SCAN)
        self.obj_centroid_z   = None   # world-z of object centroid
        self.obj_top_z        = None   # world-z of top surface
        self.obj_front_x      = None   # world-x of front face (toward robot)
        self.obj_half_x       = None   # half-extent in x
        self.obj_center_x     = None   # world-x of object center

        # Targets
        self._pos_target: np.ndarray = None    # (3,) current ball-site Cartesian target
        self._q_des: np.ndarray = None         # accumulated joint target (like kb_q_des)
        self._R_des: np.ndarray = None         # desired EE orientation (3x3), captured at phase entry

        # Push-phase bookkeeping
        self._contact_detected_push = False
        self._contact_pos_x: float  = None     # EE x when contact first occurred
        self._push_start_pos: np.ndarray = None
        self._pre_push_dwell_until: float = None   # hold still until this time before pushing

        # Squash / pull-tip bookkeeping
        self._squash_force_target = self.F_SQUASH_INIT
        self._squash_hold_start: float = None
        self._squash_pos_target: np.ndarray = None   # (3,) maintained during squash+pull
        self._slip_retries = 0

        # Slip detection window: stores (sim_time, ee_x, pitch_deg)
        self._slip_window_buf: deque = deque()

        # Statistics for summary
        self.tip_achieved = False
        self.phase_durations: dict = {}

        # ------------------------------------------------------------------
        # Data histories (appended each call to record())
        # ------------------------------------------------------------------
        self._t_hist        = []
        self._w_hist        = []
        self._quat_hist     = []
        self._ball_pose_hist= []
        self._sens_pose_hist= []
        self._con_bool_hist = []
        self._obj_pose_hist = []
        self._phase_hist    = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self):
        """
        Advance the state machine by one simulation step.
        Call this BEFORE mujoco.mj_step() each iteration.
        """
        if self.phase == Phase.IDLE:
            self._enter_phase(Phase.SCAN)

        elif self.phase == Phase.SCAN:
            self._run_scan()

        elif self.phase == Phase.APPROACH_PUSH:
            self._run_approach_push()

        elif self.phase == Phase.PUSH:
            self._run_push()

        elif self.phase == Phase.RETREAT:
            self._run_retreat()

        elif self.phase == Phase.DESCEND:
            self._run_descend()

        elif self.phase == Phase.SQUASH:
            self._run_squash()

        elif self.phase == Phase.PULL_TIP:
            self._run_pull_tip()

        elif self.phase == Phase.DONE:
            pass  # hold still

        # Safety check every step
        self._safety_check()

    def record(self):
        """Append current-timestep data to histories. Call AFTER mj_step()."""
        self._t_hist.append(self.data.time)
        self._w_hist.append(self.irb.ft_get_reading())
        self._quat_hist.append(self.irb.get_payload_pose(out='quat'))
        self._ball_pose_hist.append(self.irb.get_site_pose("ball"))
        self._sens_pose_hist.append(self.irb.get_site_pose("sensor"))
        self._con_bool_hist.append(self.irb.check_contact())
        self._obj_pose_hist.append(self.irb.get_payload_pose(out='T'))
        self._phase_hist.append(int(self.phase))

    def is_done(self) -> bool:
        return self.phase == Phase.DONE or self.irb.stop

    def save(self, path: str = "simulation_data_multiphase.npz"):
        """Convert history lists to numpy arrays and save to .npz."""
        t           = np.asarray(self._t_hist,         dtype=float)
        w           = np.asarray(self._w_hist,         dtype=float).reshape(-1, 6)
        quat        = np.asarray(self._quat_hist,      dtype=float)
        ball_pose   = np.asarray(self._ball_pose_hist, dtype=float).reshape(-1, 4, 4)
        sens_pose   = np.asarray(self._sens_pose_hist, dtype=float).reshape(-1, 4, 4)
        con_bool    = np.asarray(self._con_bool_hist,  dtype=float)
        obj_pose    = np.asarray(self._obj_pose_hist,  dtype=float).reshape(-1, 4, 4)
        phase       = np.asarray(self._phase_hist,     dtype=int)

        np.savez(
            path,
            t_hist          = t,
            w_hist          = w,
            quat_hist       = quat,
            ball_pose_hist  = ball_pose,
            sens_pose_hist  = sens_pose,
            con_bool_hist   = con_bool,
            obj_pose_hist   = obj_pose,
            ball_pos_hist   = ball_pose[:, :3, 3],
            sens_pos_hist   = sens_pose[:, :3, 3],
            obj_pos_hist    = obj_pose[:,  :3, 3],
            phase_hist      = phase,
            com_gt          = self.com_gt,
            mass_gt         = np.array([self.mass_gt]),
            mu_gt           = np.array([0.0]),   # filled externally if needed
        )
        self._log(f"[PhaseController] Saved multiphase data to {path}")

    def print_summary(self):
        """Print a human-readable summary of the run."""
        print("\n" + "=" * 60)
        print("  PhaseController Run Summary")
        print("=" * 60)
        for ph in Phase:
            if ph in self._phase_start_time:
                t0 = self._phase_start_time[ph]
                t1 = self._phase_end_time.get(ph, self.data.time)
                dur = t1 - t0
                print(f"  {ph.name:<16}  {dur:.2f} s")
        print(f"  Tip achieved    : {self.tip_achieved}")
        print(f"  Slip retries    : {self._slip_retries}")
        print("=" * 60 + "\n")

    def set_log_file(self, path: str):
        """Redirect all PhaseController print() output to `path` (and stdout).

        Must be called before the sim loop starts.  The log file is written in
        append mode so successive runs in the same session accumulate.
        """
        self._log_file = open(path, "a", buffering=1)   # line-buffered
        self._log_path = path
        self._log("=" * 60)
        self._log(f"PhaseController log  —  object {self.object_id}  —  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("=" * 60)

    def _log(self, msg: str):
        """Print to stdout and, if a log file is open, also write there."""
        print(msg)
        if self._log_file is not None:
            self._log_file.write(msg + "\n")

    def start_at_phase(self, phase: Phase):
        """Skip earlier phases and begin the state machine at `phase`.

        Runs SCAN internally (to populate object geometry) then sets the
        current phase.  The robot's physical position is NOT changed — the
        caller is responsible for placing the robot in a sensible pose for
        the chosen starting phase before calling this.

        Supported start phases: RETREAT, DESCEND, SQUASH, PULL_TIP.
        """
        if phase not in (Phase.RETREAT, Phase.DESCEND, Phase.SQUASH, Phase.PULL_TIP):
            raise ValueError(f"start_at_phase: unsupported phase {phase.name}. "
                             f"Supported: RETREAT, DESCEND, SQUASH, PULL_TIP.")

        # Always run the geometry scan so obj_top_z / obj_center_x etc. are populated.
        self._scan_object_geometry()

        # Stamp timing as if earlier phases completed instantly
        t_now = self.data.time
        for ph in Phase:
            if ph.value < phase.value:
                self._phase_start_time[ph] = t_now
                self._phase_end_time[ph]   = t_now

        self._log(f"[PhaseController] start_at_phase: jumping to {phase.name} (t={t_now:.3f} s)")
        self.phase = phase
        self._phase_start_time[phase] = t_now
        self._phase_settle_until = t_now + 0.1

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _run_scan(self):
        """Determine object geometry, then move to a safe retracted position first."""
        self._scan_object_geometry()
        self._enter_phase(Phase.APPROACH_PUSH)

    def _run_approach_push(self):
        """
        Move to the push standoff position via two safe waypoints to avoid
        colliding with the object:
          WP0: retract to safe_x (clear of the object), keep current z
          WP1: lower to centroid height at safe_x
          WP2: advance to standoff position in front of the object face
        """
        if self._pos_target is None:
            ball_pos  = self.irb.get_site_pose("ball")[:3, 3]
            # WP2 puts the ball surface just touching the front face (5 mm gap so no pre-load)
            stand_off = self.irb.ball_radius + self.PREPUSH_GAP
            # safe_x is well clear of the object in x so we can safely change height
            safe_x    = self.obj_front_x - 0.10

            self._approach_waypoints = [
                np.array([safe_x,                         0.0, ball_pos[2]]),           # WP0: retract in x
                np.array([safe_x,                         0.0, self.obj_centroid_z]),    # WP1: drop to centroid z
                np.array([self.obj_front_x - stand_off,   0.0, self.obj_centroid_z]),    # WP2: advance to face
            ]
            self._approach_wp_idx = 0
            self._pos_target = self._approach_waypoints[0]
            self._log(f"[APPROACH] Waypoints: {[list(np.round(w,3)) for w in self._approach_waypoints]}")

        self._move_toward_pos(self._pos_target, self.MOVE_SPEED)

        ball_pos = self.irb.get_site_pose("ball")[:3, 3]
        if np.linalg.norm(ball_pos - self._pos_target) < self.APPROACH_TOL:
            self._approach_wp_idx += 1
            if self._approach_wp_idx < len(self._approach_waypoints):
                self._pos_target = self._approach_waypoints[self._approach_wp_idx]
                self._log(f"[APPROACH] Waypoint {self._approach_wp_idx}: {np.round(self._pos_target, 3)}")
            else:
                self._pos_target = None
                self._contact_detected_push = False
                self._contact_pos_x = None
                self._push_start_pos = ball_pos.copy()
                self._enter_phase(Phase.PUSH)

    def _run_push(self):
        """Push forward (+x) quasi-statically, monitoring contact and force."""
        # Dwell at standoff before starting the push
        if self._pre_push_dwell_until is None:
            self._pre_push_dwell_until = self.data.time + self.PRE_PUSH_DWELL
            self._log(f"[PUSH] Dwelling at standoff for {self.PRE_PUSH_DWELL} s...")
        if self.data.time < self._pre_push_dwell_until:
            if self._q_des is not None:
                self.irb.set_pos_ctrl(self._q_des, check_ellipsoid=False)
            return

        ee_pos = self.irb.get_site_pose("ball")[:3, 3]
        ft     = self.irb.ft_get_reading()
        f_mag  = np.linalg.norm(ft[:3])

        # Detect contact onset
        if not self._contact_detected_push:
            if f_mag > self.CONTACT_FORCE_THRESH or self.irb.check_contact():
                self._contact_detected_push = True
                self._contact_pos_x = ee_pos[0]
                self._log(f"[PUSH] Contact detected at x={ee_pos[0]:.4f} m, |F|={f_mag:.2f} N")

        # Check push distance after contact
        if self._contact_detected_push and self._contact_pos_x is not None:
            push_dist = ee_pos[0] - self._contact_pos_x
            if push_dist >= self.PUSH_DIST_AFTER_CONTACT:
                self._log(f"[PUSH] Push distance reached ({push_dist*100:.1f} cm). Transitioning.")
                self._enter_phase(Phase.RETREAT)
                return

        # Check tip onset
        pitch_deg = self._get_obj_pitch_deg()
        if abs(pitch_deg) > self.TIP_DETECT_DEG:
            self._log(f"[PUSH] Tip/slide detected (pitch={pitch_deg:.1f}°). Transitioning.")
            self._enter_phase(Phase.RETREAT)
            return

        # Force safety limit
        if f_mag > self.PUSH_FORCE_LIMIT:
            self._log(f"[PUSH] Force limit {self.PUSH_FORCE_LIMIT} N reached. Transitioning.")
            self._enter_phase(Phase.RETREAT)
            return

        # Step forward in +x at push speed, locked to centroid height
        new_pos = np.array([ee_pos[0] + 1.0, 0.0, self.obj_centroid_z])
        self._move_toward_pos(new_pos, self.PUSH_SPEED_CTRL)

    def _run_retreat(self):
        """
        Three-waypoint retreat using ball-site positions:
          1. Pull back in -x to clear the object
          2. Rise +z above the object top
          3. Advance +x to be directly above the object center
        Re-observes object position at waypoint init so post-push displacement is accounted for.
        """
        if self._pos_target is None:
            ball_pos = self.irb.get_site_pose("ball")[:3, 3]
            self._update_obj_geometry()   # re-read object position after push
            self._retreat_waypoints = self._compute_retreat_waypoints(ball_pos)
            self._retreat_wp_idx = 0
            self._pos_target = self._retreat_waypoints[0]
            self._log(f"[RETREAT] Object re-observed: top_z={self.obj_top_z:.3f}, center_x={self.obj_center_x:.3f}")

        self._move_toward_pos(self._pos_target, self.MOVE_SPEED)

        ball_pos = self.irb.get_site_pose("ball")[:3, 3]
        if np.linalg.norm(ball_pos - self._pos_target) < self.APPROACH_TOL:
            self._retreat_wp_idx += 1
            if self._retreat_wp_idx < len(self._retreat_waypoints):
                self._pos_target = self._retreat_waypoints[self._retreat_wp_idx]
            else:
                self._pos_target = None
                self._enter_phase(Phase.DESCEND)

    def _run_descend(self):
        """Move EE downward until contact with object top is detected."""
        # Ignore force readings for the first 200 ms — residual squash force from
        # the retreat move triggers a false contact detection immediately otherwise.
        if self.data.time < self._phase_settle_until + 0.1:
            ball_pos = self.irb.get_site_pose("ball")[:3, 3]
            target   = ball_pos.copy()
            target[2] -= 1.0
            self._move_toward_pos(target, self.DESCEND_SPEED_CTRL)
            return

        ft  = self._get_ft_world()
        f_z = abs(ft[2])   # world-z: vertical force

        if f_z > self.DESCEND_CONTACT_F or self.irb.check_contact():
            self._log(f"[DESCEND] Contact on top surface. fz={f_z:.2f} N")
            ball_pos = self.irb.get_site_pose("ball")[:3, 3]
            self._squash_pos_target = ball_pos.copy()
            self._squash_hold_start = None
            self._enter_phase(Phase.SQUASH)
            return

        # Step downward — target far below, speed caps actual motion
        ball_pos = self.irb.get_site_pose("ball")[:3, 3]
        target   = ball_pos.copy()
        target[2] -= 1.0   # direction only
        self._move_toward_pos(target, self.DESCEND_SPEED_CTRL)

    def _run_squash(self):
        """Proportional force control: move down until F_z target is reached and held."""
        ft  = self._get_ft_world()
        f_z = abs(ft[2])   # world-z: vertical force

        ball_pos = self.irb.get_site_pose("ball")[:3, 3]
        self._squash_pos_target = ball_pos.copy()

        # Proportional descent speed: fast when far from target, slows as force builds
        force_error = self._squash_force_target - f_z
        if force_error > 0:
            # Scale speed linearly with error: full speed when error = target, zero when error = 0
            squash_speed = self.SQUASH_SPEED_MAX * (force_error / self._squash_force_target)
            squash_speed = max(squash_speed, 0.0005)   # floor to avoid stall
            target = ball_pos.copy()
            target[2] -= 1.0
            self._move_toward_pos(target, squash_speed)
        else:
            # Over target — hold current joint position
            if self._q_des is not None:
                self.irb.set_pos_ctrl(self._q_des, check_ellipsoid=False)

        # Track hold time once force is close enough (within 20%)
        if f_z >= self._squash_force_target * 0.8:
            if self._squash_hold_start is None:
                self._squash_hold_start = self.data.time
                self._log(f"[SQUASH] Force target reached ({f_z:.2f}/{self._squash_force_target:.2f} N). Holding...")
            elif (self.data.time - self._squash_hold_start) >= self.SQUASH_HOLD_TIME:
                self._log(f"[SQUASH] Hold complete. Starting pull-tip.")
                self._slip_window_buf.clear()
                self._enter_phase(Phase.PULL_TIP)
        else:
            self._squash_hold_start = None  # reset if force drops out

    def _run_pull_tip(self):
        """Lateral pull with maintained squash force; detect tip success or slip."""
        ft  = self._get_ft_world()
        f_z = abs(ft[2])   # world-z: vertical force
        dt  = float(self.model.opt.timestep)

        # _q_des and _q_squash carry over from SQUASH — do NOT reinitialise here.

        # During stabilisation window: hold completely still and let PD settle.
        stabilising = self.data.time < self._pull_stable_until
        if stabilising:
            self.irb.set_pos_ctrl(self._q_des, check_ellipsoid=False)
            return

        # --- Lateral pull only ---
        # Accumulate only the lateral (+x) velocity into _q_des.
        # Z is handled separately below to avoid kinematic coupling lifting the EE.
        v_cmd = np.zeros(6)
        v_cmd[3] = -self.PULL_SPEED   # vx only

        self.irb.get_jacobian(set_pinv=True)
        q_dot = self.irb.J_pinv @ v_cmd
        q_dot = np.clip(q_dot, -self.irb.v_max, self.irb.v_max)
        self._q_des = self._q_des + q_dot * dt
        self._q_des = np.clip(self._q_des, self.irb.q_min, self.irb.q_max)

        # --- Z floor: never let joints go shallower than squash depth ---
        # If force drops, clamp each joint back to its squash value (deepest position).
        # This prevents kinematic coupling from lifting the finger off the object.
        if f_z < self._squash_force_target * 0.5 and self._q_squash is not None:
            # Re-apply squash depth by taking the element-wise value that keeps
            # the robot deeper. For joints that move the EE down when increased,
            # the squash value is the one that produced contact — just restore it.
            self._log(f"[PULL_TIP] Force dropped to {f_z:.2f} N — restoring squash depth.")
            self._q_des = self._q_squash.copy()

        self.irb.set_pos_ctrl(self._q_des, check_ellipsoid=False)

        # --- Tip detection ---
        pitch_deg = self._get_obj_pitch_deg()

        if abs(pitch_deg) > self.TIP_DONE_DEG:
            self._log(f"[PULL_TIP] Tipping complete (pitch={pitch_deg:.1f}°). Done.")
            self.tip_achieved = True
            self._enter_phase(Phase.DONE)
            return

        # --- Slip detection ---
        ee_pos = self.irb.get_site_pose("ball")[:3, 3]
        now    = self.data.time

        # Record pull start position on first active (non-stabilising) step
        if self._pull_start_x is None:
            self._pull_start_x = ee_pos[0]

        total_travel = abs(ee_pos[0] - self._pull_start_x)

        self._slip_window_buf.append((now, ee_pos[0], pitch_deg))

        # Prune entries older than window
        while self._slip_window_buf and (now - self._slip_window_buf[0][0]) > self.SLIP_WINDOW:
            self._slip_window_buf.popleft()

        # Only evaluate slip after minimum lateral travel — avoids false positives
        # during the stabilisation window and initial contact transients.
        if total_travel >= self.SLIP_MIN_TRAVEL and len(self._slip_window_buf) >= 2:
            t_old, x_old, p_old = self._slip_window_buf[0]
            delta_x_ee  = abs(ee_pos[0] - x_old)
            delta_pitch = abs(pitch_deg - p_old)

            if delta_x_ee > self.SLIP_EE_THRESH and delta_pitch < self.SLIP_PITCH_THRESH:
                self._log(f"[PULL_TIP] Slip detected! total_travel={total_travel*1000:.1f} mm, "
                      f"Δx_ee={delta_x_ee*1000:.1f} mm, Δpitch={delta_pitch:.2f}°")
                self._handle_slip()

    def _handle_slip(self):
        """Increment squash force and retry from RETREAT, or give up."""
        self._slip_retries += 1
        if self._slip_retries > self.MAX_SLIP_RETRIES:
            self._log(f"[PULL_TIP] Max slip retries ({self.MAX_SLIP_RETRIES}) reached. Stopping.")
            self._enter_phase(Phase.DONE)
            return

        new_f = min(self._squash_force_target * 1.5, self.F_SQUASH_MAX)
        self._log(f"[PULL_TIP] Retry {self._slip_retries}/{self.MAX_SLIP_RETRIES}. "
              f"Increasing squash force: {self._squash_force_target:.1f} → {new_f:.1f} N")
        self._squash_force_target = new_f
        self._squash_hold_start   = None
        self._pos_target          = None
        self._q_target            = None
        self._enter_phase(Phase.RETREAT)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _update_obj_geometry(self):
        """Re-read object AABB from current MuJoCo state (call after push to get updated position)."""
        self._scan_object_geometry(verbose=False)

    def _scan_object_geometry(self, verbose=True):
        """
        Determine object bounding box geometry from MuJoCo geom data.
        Works for both primitive (box) and mesh geoms by reading geom aabb.
        """
        # Find the payload body
        payload_body_id = self.irb.payload_body_id

        # Collect all geoms belonging to the payload body
        geom_ids = [
            g for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) == payload_body_id
        ]

        # For each geom, get its world-frame AABB by reading pos + size
        # We only care about collision geoms (not sites); type 5 = mjGEOM_MESH, 6 = mjGEOM_BOX, etc.
        # Ignore geom type 7 (mjGEOM_NONE) and purely visual ones with contype/condim=0
        all_min = []
        all_max = []

        mujoco.mj_forward(self.model, self.data)

        for gid in geom_ids:
            gtype = int(self.model.geom_type[gid])
            # Skip visual-only geoms (condim==0 or contype==0)
            if self.model.geom_contype[gid] == 0:
                continue

            # World-frame geom position
            gpos = self.data.geom_xpos[gid].copy()

            # Get half-extents depending on type
            sz = self.model.geom_size[gid].copy()
            if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                half = sz[:3]
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                half = np.array([sz[0], sz[0], sz[0]])
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                half = np.array([sz[0], sz[0], sz[1]])
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                half = np.array([sz[0], sz[0], sz[1] + sz[0]])
            else:
                # For mesh geoms: use the bounding box stored in model.mesh_vert
                # Fall back to a conservative estimate from geom_rbound
                r = float(self.model.geom_rbound[gid])
                half = np.array([r, r, r])

            all_min.append(gpos - half)
            all_max.append(gpos + half)

        if not all_min:
            # Fallback: use payload site position and a 0.1 m cube
            p = self.irb.get_payload_pose(out='p')
            self._log("[SCAN] Warning: no collision geoms found, using fallback geometry.")
            all_min = [p - 0.1]
            all_max = [p + 0.1]

        aabb_min = np.min(np.stack(all_min, axis=0), axis=0)
        aabb_max = np.max(np.stack(all_max, axis=0), axis=0)

        # Table surface z
        table_z = self.irb.get_surface_pos()[2]

        self.obj_top_z      = float(aabb_max[2])
        self.obj_centroid_z = float((aabb_min[2] + aabb_max[2]) / 2.0)
        self.obj_front_x    = float(aabb_min[0])   # min x = face closest to robot (robot is at -x)
        self.obj_center_x   = float((aabb_min[0] + aabb_max[0]) / 2.0)
        self.obj_half_x     = float((aabb_max[0] - aabb_min[0]) / 2.0)

        if verbose:
            self._log(f"[SCAN] Object geometry: "
                  f"top_z={self.obj_top_z:.3f}, centroid_z={self.obj_centroid_z:.3f}, "
                  f"front_x={self.obj_front_x:.3f}, center_x={self.obj_center_x:.3f}")

    def _compute_retreat_waypoints(self, ee_pos: np.ndarray) -> list:
        """
        Return a list of (3,) positions describing the retreat path:
          1. Pull back in -x to clear the front face
          2. Rise to above the object top
          3. Move forward to be above the object center
        """
        clearance_x    = self.obj_front_x - self.RETREAT_CLEARANCE
        above_z        = self.obj_top_z + self.TOP_CLEARANCE + self.irb.ball_radius

        wp1 = np.array([clearance_x,     0.0, ee_pos[2]])      # pull back at same height
        wp2 = np.array([clearance_x,     0.0, above_z])         # rise up
        wp3 = np.array([self.obj_center_x, 0.0, above_z])       # move over object center

        return [wp1, wp2, wp3]

    def _get_obj_pitch_deg(self) -> float:
        """Return the object's pitch angle about the Y-axis in degrees."""
        rpy = self.irb.get_payload_pose(out='rpy', degrees=True)
        return float(rpy[1])   # pitch = rotation about Y

    def _get_ft_world(self) -> np.ndarray:
        """Return the F/T reading rotated into the world frame.

        ft_get_reading() returns forces in the sensor's local frame.  Rotating
        by R_sensor (sensor axes expressed in world) gives world-frame forces,
        so ft_world[2] is always the vertical (world-z) component regardless of
        wrist orientation.
        """
        ft_sensor = self.irb.ft_get_reading()
        R_sensor  = self.data.site_xmat[self.irb.ft_site].reshape(3, 3)
        f_world   = R_sensor @ ft_sensor[:3]
        t_world   = R_sensor @ ft_sensor[3:]
        return np.concatenate([f_world, t_world])

    # ------------------------------------------------------------------
    # Control helpers
    # ------------------------------------------------------------------

    def _ik_for_ball_pos(self, ball_target_pos: np.ndarray) -> np.ndarray:
        """
        Compute IK so that the BALL SITE reaches ball_target_pos, keeping current orientation.

        irb.IK() minimises error at site:tool0 (EE flange), not site:ball_center.
        We correct for the fixed ball→flange offset measured in the current world frame.
        Returns joint angles (6,).
        """
        mujoco.mj_forward(self.model, self.data)
        T_ee   = self.irb.FK()                          # flange pose
        T_ball = self.irb.get_site_pose("ball")         # ball-center pose

        # Vector from ball to EE flange, in world frame (constant for fixed orientation)
        ball_to_ee = T_ee[:3, 3] - T_ball[:3, 3]

        # Target flange position that puts the ball at ball_target_pos
        flange_target = ball_target_pos + ball_to_ee

        T_target = T_ee.copy()
        T_target[:3, 3] = flange_target
        try:
            q = self.irb.IK(T_target, method=2, damping=0.5, max_iters=500)
        except RuntimeError:
            q = self.data.qpos[self.irb.joint_idx].copy()
        return q

    def _move_toward_pos(self, ball_target_pos: np.ndarray, speed: float):
        """
        Issue one position-control command that steps the ball site toward
        ball_target_pos at `speed` m/s.

        Mirrors apply_cartesian_keyboard_ctrl: integrates q_dot into a persistent
        _q_des accumulator so the PD controller always sees a nonzero error and
        produces enough torque to actually move the robot.
        """
        dt = float(self.model.opt.timestep)
        mujoco.mj_forward(self.model, self.data)

        # Initialise persistent joint target and desired orientation on first call
        if self._q_des is None:
            self._q_des = self.data.qpos[self.irb.joint_idx].copy().astype(float)
        if self._R_des is None:
            self._R_des = self.irb.FK()[:3, :3].copy()

        ball_pos = self.irb.get_site_pose("ball")[:3, 3]
        diff     = ball_target_pos - ball_pos
        dist     = np.linalg.norm(diff)

        # Linear velocity toward target, capped at speed
        v_lin = (diff / dist) * speed if dist > 1e-6 else np.zeros(3)

        # Orientation error: w_err = R_des * log(R_des^T * R_curr) expressed in world frame
        R_curr  = self.irb.FK()[:3, :3]
        R_err   = self._R_des @ R_curr.T          # rotation that takes R_curr → R_des
        # Convert to axis-angle (rotation vector) — this is the angular velocity needed
        rotvec  = R.from_matrix(R_err).as_rotvec()
        w_ori   = self.ORI_KP * rotvec             # proportional angular correction (rad/s)

        # Full 6D twist: [wx wy wz vx vy vz]
        v_cmd = np.zeros(6)
        v_cmd[:3] = w_ori
        v_cmd[3:] = v_lin

        # Differential IK: accumulate into persistent joint target
        self.irb.get_jacobian(set_pinv=True)
        q_dot = self.irb.J_pinv @ v_cmd
        q_dot = np.clip(q_dot, -self.irb.v_max, self.irb.v_max)

        self._q_des = self._q_des + q_dot * dt
        self._q_des = np.clip(self._q_des, self.irb.q_min, self.irb.q_max)
        self.irb.set_pos_ctrl(self._q_des, check_ellipsoid=False)

    # ------------------------------------------------------------------
    # Phase transition
    # ------------------------------------------------------------------

    def _enter_phase(self, new_phase: Phase):
        t_now = self.data.time
        if self.phase in self._phase_start_time:
            self._phase_end_time[self.phase] = t_now
        self._phase_start_time[new_phase] = t_now
        self._phase_settle_until = t_now + 0.1   # ignore force safety for 100 ms after transition
        if new_phase == Phase.PULL_TIP:
            self._pull_stable_until = t_now + 0.3  # hold z-only for 300 ms before lateral pull
            self._q_squash = self._q_des.copy() if self._q_des is not None else None
            self._pull_start_x = None   # will be set on first active pull step
        self._log("")
        self._log(f"[PhaseController] {self.phase.name} → {new_phase.name}  (t={t_now:.3f} s)")
        self.phase = new_phase

        # Clear waypoint target so the new phase recomputes its own goals
        if new_phase in (Phase.APPROACH_PUSH, Phase.RETREAT):
            self._pos_target = None

        # Keep accumulated joint target across smooth motion chains:
        #   APPROACH_PUSH → PUSH  (continue forward)
        #   SQUASH → PULL_TIP     (preserve squash depth so force is not released)
        # All other transitions reset so the new phase starts from actual robot state.
        if new_phase not in (Phase.PUSH, Phase.PULL_TIP):
            self._q_des  = None
            self._R_des  = None  # capture fresh orientation reference at next move call

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def _safety_check(self):
        """Emergency stop conditions checked every step."""
        if self.phase == Phase.DONE:
            return

        # Skip force check during the settling window after a phase transition —
        # inertial transients from fast moves cause spurious force spikes.
        if self.data.time < self._phase_settle_until:
            return

        ft    = self._get_ft_world()
        f_mag = np.linalg.norm(ft[:3])
        if f_mag > self.SAFETY_FORCE_LIMIT:
            self._log(f"[SAFETY] Force magnitude {f_mag:.1f} N exceeds limit. Stopping.")
            self._enter_phase(Phase.DONE)
            self.irb.stop = True
            return

        pitch = abs(self._get_obj_pitch_deg())
        if pitch > self.TIP_ANGLE_ABORT and self.phase not in (Phase.PULL_TIP,):
            self._log(f"[SAFETY] Object pitch {pitch:.1f}° > {self.TIP_ANGLE_ABORT}°. Stopping.")
            self._enter_phase(Phase.DONE)

    # ------------------------------------------------------------------
    # Parameter loading
    # ------------------------------------------------------------------

    def _load_params(self, object_id: int):
        with open(_PARAMS_FILE, "r") as f:
            params = json.load(f)["objects"][str(object_id)]
        self.com_gt  = np.subtract(params["com_gt_onshape"], params["com_gt_offset"])
        self.mass_gt = float(params["mass_gt"])
        self.init_xyz = np.array(params["init_xyz"])
        self._log(f"[PhaseController] Object {object_id} ({params['name']}): "
              f"mass={self.mass_gt:.3f} kg, com_gt={self.com_gt}")
