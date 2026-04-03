"""
phase_controller.py
-------------------
Shared multi-phase state machine for the ABB IRB120 robot in MuJoCo.

This module owns the phase ordering, geometry extraction, logging, slip/tip
detection, and other helper logic. Actuator-specific motion commands are
implemented by subclasses in phase_controllers.py.

State machine:
    IDLE → SCAN → APPROACH_PUSH → PUSH → RETREAT_TO_TOP → DESCEND → SQUASH → PULL_TIP → RETURN_PRE_SQUASH → DONE

Usage (from sim loop):
    pc = StateMachine(irb, model, data, object_id=0)
    while not pc.is_done():
        pc.step()
        mujoco.mj_step(model, data)
        pc.record()
    pc.save("simulation_data_multiphase.npz")
"""

import json
import time
from datetime import date
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
    RETURN_PRE_SQUASH = 8
    DONE          = 9


PHASE_NAMES = {p: p.name for p in Phase}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT  = Path(__file__).resolve().parents[3]
_PARAMS_FILE = _REPO_ROOT / "src" / "mujoco_irb120" / "assets" / "object_params.json"


# ---------------------------------------------------------------------------
# StateMachine
# ---------------------------------------------------------------------------

class StateMachine:
    """
    Autonomous state-machine controller that sequences through multiple
    interaction phases (push, squash, pull-tip) to collect data for
    joint mass/CoM/friction estimation.

    Args:
        irb         : robot.Robot instance
        model       : mujoco.MjModel
        data        : mujoco.MjData
        object_id   : integer key into object_params.json (default 0 = box)
    """

    # ------------------------------------------------------------------
    # Tunable constants
    # ------------------------------------------------------------------
    # Approach / push
    # PUSH_SPEED          = 0.03      # m/s forward push speed
    PUSH_DIST_AFTER_CONTACT = 0.04  # how far to push after contact (m)
    PUSH_FORCE_LIMIT    = 20.0      # N — abort push if exceeded
    SAFETY_FORCE_LIMIT  = 30.0      # N — emergency retreat
    TIP_ANGLE_ABORT     = 60.0      # deg — stop if object tips this far
    TIP_DETECT_DEG      = 2.0       # deg — tip detected during push
    CONTACT_FORCE_THRESH= 0.5       # N — force magnitude for contact onset
    APPROACH_TOL        = 0.005     # m — position tolerance to advance phase
    PREPUSH_GAP         = 0.01      # 0.005 m — initial gap to object before pushing (no pre-load)
    PRE_PUSH_DWELL      = 1.0       # s — pause at standoff before starting push

    # Retreat / top
    RETREAT_CLEARANCE   = 0.06      # m — pull back from object in -x before going up
    TOP_CLEARANCE       = 0.03      # m — how far above object top to hover

    # Descend
    # DESCEND_SPEED       = 0.02      # 0.01 m/s
    DESCEND_CONTACT_F   = 0.5       # N — contact force for descent stop

    # Squash / force control
    F_SQUASH_INIT       = 8.0      # 15.0 6.0 N initial squash force target
    F_SQUASH_MAX        = 12.0      # 20.0 16 N cap on retried squash force
    # F_SQUASH_KP         = 0.0001    # m/N proportional gain
    SQUASH_HOLD_TIME    = 0.1       # 0.5 s — hold at target before transitioning

    # Pull-tip
    PULL_FZ_KP          = 0.003     # m/s per N — z velocity gain for force regulation during pull
    PULL_Z_VEL_MAX      = 0.004     # m/s cap for z correction while pulling
    # TIP_SUCCESS_DEG     = 5.0       # deg pitch to declare tipping started
    TIP_DONE_DEG        = 10.0      # deg pitch to stop pulling
    SLIP_WINDOW         = 0.5       # s sliding window for slip detection
    SLIP_EE_THRESH      = 0.002     # m EE lateral movement within window to flag slip
    SLIP_PITCH_THRESH   = 0.5       # deg pitch change below which slip declared
    SLIP_MIN_TRAVEL     = 0.03      # 0.01 m minimum total lateral travel before slip can be declared
    MAX_SLIP_RETRIES    = 1         # +1 = 2 total attempts at squash-pull

    # Speed limits for quasi-static motion
    MOVE_SPEED          = 0.08      # m/s — max Cartesian speed for approach / retreat moves
    PUSH_SPEED_CTRL     = 0.03      # m/s — push speed
    DESCEND_SPEED_CTRL  = 0.02      # m/s — descend speed (slow to avoid overshoot on contact)
    SQUASH_SPEED_MIN    = 0.035    # 0.0005 m/s — never descend slower than this while force is still rising
    SQUASH_SPEED_MAX    = 0.005    # 0.0025 m/s — max speed during squash force control
    SQUASH_FORCE_WINDOW = 6        # samples used to smooth force before declaring squash complete
    SQUASH_FORCE_READY_RATIO = 0.98  # filtered force must reach this fraction of target before hold
    PULL_SPEED          = 0.04      # 0.02 m/s lateral pull speed
    RETURN_Q_SPEED      = 0.20      # rad/s per-joint cap for gentle post-pull return
    RETURN_Q_TOL        = 0.02      # rad joint-space tolerance for return completion
    RETURN_PATH_SPEED   = 0.015     # m/s waypoint tracking speed for pull-path replay
    RETURN_WP_TOL       = 0.003     # m waypoint reach tolerance during return
    RETURN_PATH_STRIDE  = 10        # decimation stride for recorded pull path samples
    RETURN_ARC_Z        = 0.012     # m max upward arc offset while returning
    RETURN_FZ_TARGET    = 1.5       # N final normal-force target near return completion
    RETURN_FZ_START     = 4.0       # N initial normal-force target at return start
    RETURN_FZ_KP        = 0.002     # m/s per N, upward force-relief gain in return
    RETURN_FZ_VMAX      = 0.006     # m/s cap on upward relief velocity
    RETURN_FZ_DOWN_VMAX = 0.0015    # m/s cap on downward correction (keep gentle on unstable object)
    RETURN_FORCE_RECOVER_THRESH = 0.6  # below this ratio of target force, slow lateral replay
    RETURN_FORCE_SLOW_RATIO = 0.35  # lateral speed scale when contact force is weak
    RETURN_USE_REVERSE_PULL = True   # if True, return by inverting pull direction with same force loop
    RETURN_X_TOL = 0.002             # m tolerance to stop reverse-pull at pull anchor x
    ORI_KP              = 2.0       # rad/s per rad of orientation error — restores EE orientation

    def __init__(self, irb, model: mujoco.MjModel, data: mujoco.MjData, object_id: int = 0):
        self.irb    = irb
        self.model  = model
        self.data   = data
        self.object_id = object_id

        # Logging (must be initialized before _load_params which calls _log)
        self._log_file = None
        self._log_path = None
        self._log_date = None

        # Load ground-truth params
        self._load_params(object_id)

        # Current phase
        self.phase = Phase.IDLE

        # Phase timing
        self._phase_start_time: dict[Phase, float] = {}
        self._phase_end_time:   dict[Phase, float] = {}
        self._phase_settle_until: float = 0.0   # ignore safety check until this sim time
        self._pull_stable_until: float = 0.0   # PULL_TIP: hold z before lateral motion
        self._pull_start_x: float = None       # ball x at start of lateral pull (for min-travel gate)
        self._q_squash: np.ndarray = None      # joint target captured at squash completion (z floor)
        self._q_pre_squash: np.ndarray = None  # joint target captured at SQUASH entry (pre-load state)
        self._pull_anchor_pos: np.ndarray = None   # ball position at PULL_TIP entry (start of pull)
        self._pull_path_pos: list[np.ndarray] = [] # sampled ball positions during pull, for replay return
        self._return_path: list[np.ndarray] = []   # reverse pull waypoints with arc lift
        self._return_wp_idx: int = 0
        self._return_fz_start: float = None        # measured Fz at return entry for target ramp

        # Geometry info (filled during SCAN)
        self.obj_centroid_z   = None   # world-z of object centroid
        self.obj_top_z        = None   # world-z of top surface
        self.obj_front_x      = None   # world-x of front face (toward robot)
        self.obj_half_x       = None   # half-extent in x
        self.obj_center_x     = None   # world-x of object center
        self.obj_squash_x     = None   # world-x target for squash: front edge + ball_radius

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
        self._squash_force_ready_since: float = None
        self._squash_fz_window: deque = deque(maxlen=self.SQUASH_FORCE_WINDOW)
        self._squash_pos_target: np.ndarray = None   # (3,) maintained during squash+pull
        self._slip_retries = 0

        # Slip detection window: stores (sim_time, ee_x, pitch_deg)
        self._slip_window_buf: deque = deque()
        self._slip_lift: bool = False   # when True, RETREAT lifts straight up instead of retracting in x

        # Statistics for summary
        self.tip_achieved = False
        self.phase_durations: dict = {}

        # ------------------------------------------------------------------
        # Data histories (appended each call to record())
        # ------------------------------------------------------------------
        self._t_hist        = []
        self._w_sensor_hist = []
        self._w_world_hist  = []
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

        elif self.phase == Phase.RETURN_PRE_SQUASH:
            self._run_return_pre_squash()

        elif self.phase == Phase.DONE:
            pass  # hold still

        # Safety check every step
        self._safety_check()

    def record(self):
        """Append current-timestep data to histories. Call AFTER mj_step()."""
        self._t_hist.append(self.data.time)
        self._w_sensor_hist.append(self.irb.ft_get_reading(flip_sign=True))
        self._w_world_hist.append(self._get_ft_world())
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
        w_sensor    = np.asarray(self._w_sensor_hist,         dtype=float).reshape(-1, 6)
        w_world     = np.asarray(self._w_world_hist,          dtype=float).reshape(-1, 6)
        quat        = np.asarray(self._quat_hist,      dtype=float)
        ball_pose   = np.asarray(self._ball_pose_hist, dtype=float).reshape(-1, 4, 4)
        sens_pose   = np.asarray(self._sens_pose_hist, dtype=float).reshape(-1, 4, 4)
        con_bool    = np.asarray(self._con_bool_hist,  dtype=float)
        obj_pose    = np.asarray(self._obj_pose_hist,  dtype=float).reshape(-1, 4, 4)
        phase       = np.asarray(self._phase_hist,     dtype=int)

        np.savez(
            path,
            t_hist          = t,
            w_sensor_hist   = w_sensor,
            w_world_hist    = w_world,
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

        Must be called before the sim loop starts. A date-stamped log file is
        created per day, and repeated runs on the same date append to that file.
        """
        log_date = date.today().isoformat()
        log_path = Path(path)
        dated_name = f"{log_path.stem}_{log_date}{log_path.suffix or '.log'}"
        dated_path = log_path.with_name(dated_name)

        self._log_file = open(dated_path, "a", buffering=1)   # line-buffered
        self._log_path = str(dated_path)
        self._log_date = log_date
        self._log("=" * 60)
        self._log(f"PhaseController log  —  object {self.object_id}  —  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("=" * 60)
        return self._log_path

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
        Move to the push standoff position via three safe waypoints:
          WP0: retract to safe_x (clear of the object), keep current z  — MOVE_SPEED
          WP1: lower to centroid height at safe_x                        — MOVE_SPEED
          WP2: slow advance to standoff in front of the object face      — PUSH_SPEED_CTRL
        WP2 uses push speed so the robot arrives gently with no overshoot,
        ensuring the dwell pause fires before any contact.
        """
        if self._pos_target is None:
            ball_pos  = self.irb.get_site_pose("ball")[:3, 3]
            # Standoff: ball surface 2 cm from face — enough clearance to stop cleanly
            stand_off = self.irb.ball_radius + self.PREPUSH_GAP
            safe_x    = self.obj_front_x - 0.10

            self._approach_waypoints = [
                np.array([safe_x,                        0.0, ball_pos[2]]),           # WP0: retract in x
                np.array([safe_x,                        0.0, self.obj_centroid_z]),    # WP1: drop to centroid z
                np.array([self.obj_front_x - stand_off,  0.0, self.obj_centroid_z]),    # WP2: slow advance to face
            ]
            # Per-waypoint speeds: fast for repositioning, slow for final approach
            self._approach_wp_speeds = [self.MOVE_SPEED, self.MOVE_SPEED, self.PUSH_SPEED_CTRL]
            self._approach_wp_idx = 0
            self._pos_target = self._approach_waypoints[0]
            waypoints_log = [np.round(np.asarray(w, dtype=float), 3).tolist() for w in self._approach_waypoints]
            self._log(f"[APPROACH] Waypoints: {waypoints_log}")

        speed = self._approach_wp_speeds[self._approach_wp_idx]
        self._move_toward_pos(self._pos_target, speed)

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
            self._hold_still()
            return

        ee_pos = self.irb.get_site_pose("ball")[:3, 3]
        ft     = self.irb.ft_get_reading(flip_sign=True)
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
        Retreat using ball-site positions.

        Normal (post-push) path — 3 waypoints:
          1. Pull back in -x to clear the object
          2. Rise +z above the object top
          3. Advance +x to be directly above the object center

        Slip-recovery path — 2 waypoints (lift only, no x-retract):
          1. Rise straight up to above the object top (same x)
          2. Advance +x to be directly above the object center

        Re-observes object position at waypoint init.
        """
        if self._pos_target is None:
            ball_pos = self.irb.get_site_pose("ball")[:3, 3]
            self._update_obj_geometry()   # re-read object position after push/slip
            above_z = self.obj_top_z + self.TOP_CLEARANCE + self.irb.ball_radius
            if self._slip_lift:
                # Lift straight up from current x — avoid dragging finger across object
                self._retreat_waypoints = [
                    np.array([ball_pos[0],         0.0, above_z]),            # rise in place
                    np.array([self.obj_squash_x,   0.0, above_z]),            # move over front edge
                ]
                self._slip_lift = False   # consume the flag
                self._log(f"[RETREAT] Slip-recovery lift: rise to z={above_z:.3f}, then over squash_x={self.obj_squash_x:.3f}")
            else:
                self._retreat_waypoints = self._compute_retreat_waypoints(ball_pos)
                self._log(f"[RETREAT] Object re-observed: top_z={self.obj_top_z:.3f}, center_x={self.obj_center_x:.3f}")
            self._retreat_wp_idx = 0
            self._pos_target = self._retreat_waypoints[0]

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

        self._squash_fz_window.append(f_z)
        f_z_filt = float(np.mean(self._squash_fz_window))

        ball_pos = self.irb.get_site_pose("ball")[:3, 3]
        self._squash_pos_target = ball_pos.copy()

        # Proportional descent speed: fast when far from target, slows as force builds
        force_error = self._squash_force_target - f_z_filt
        if force_error > 0:
            # Normalize the force error and map it onto a bounded speed range.
            force_ratio = np.clip(force_error / max(self._squash_force_target, 1e-6), 0.0, 1.0)
            squash_speed = self.SQUASH_SPEED_MIN #+ (self.SQUASH_SPEED_MAX - self.SQUASH_SPEED_MIN) * force_ratio
            target = ball_pos.copy()
            target[2] -= 1.0
            self._move_toward_pos(target, squash_speed)
        else:
            # Over target — hold current joint position
            self._hold_still()

        # Only declare success after a short window of sustained force so a
        # single transient spike cannot trigger the hold / pull-tip transition.
        hold_ready = (
            len(self._squash_fz_window) == self.SQUASH_FORCE_WINDOW and
            f_z_filt >= self._squash_force_target * self.SQUASH_FORCE_READY_RATIO
        )
        if hold_ready:
            if self._squash_force_ready_since is None:
                self._squash_force_ready_since = self.data.time
                self._squash_hold_start = self.data.time
                self._log(
                    f"[SQUASH] Stable force reached (raw={f_z:.2f} N, filt={f_z_filt:.2f} N, "
                    f"target={self._squash_force_target:.2f} N). Holding..."
                )
            elif (self.data.time - self._squash_force_ready_since) >= self.SQUASH_HOLD_TIME:
                self._log(f"[SQUASH] Hold complete. Starting pull-tip.")
                self._slip_window_buf.clear()
                self._enter_phase(Phase.PULL_TIP)
        else:
            self._squash_force_ready_since = None
            self._squash_hold_start = None  # reset if force drops out

    def _run_pull_tip(self):
        """Lateral pull with maintained squash force; detect tip success or slip."""
        ft  = self._get_ft_world()
        f_z = abs(ft[2])   # world-z: vertical force

        # _q_des and _q_squash carry over from SQUASH — do NOT reinitialise here.

        # During stabilisation window: hold completely still and let PD settle.
        stabilising = self.data.time < self._pull_stable_until
        if stabilising:
            self._hold_still()
            return

        # --- Lateral pull + z force regulation ---
        # Keep pulling in x while adjusting z to maintain squash force.
        # v_cmd ordering is [wx wy wz vx vy vz].
        self._ensure_desired_orientation()
        fz_err = self._squash_force_target - f_z
        vz_cmd = -self.PULL_FZ_KP * fz_err
        vz_cmd = float(np.clip(vz_cmd, -self.PULL_Z_VEL_MAX, self.PULL_Z_VEL_MAX))

        v_cmd = np.zeros(6)
        v_cmd[3] = -self.PULL_SPEED   # vx only
        v_cmd[5] = vz_cmd             # regulate vertical force
        self._apply_cartesian_twist(v_cmd)

        # --- Tip detection ---
        pitch_deg = self._get_obj_pitch_deg()

        if abs(pitch_deg) > self.TIP_DONE_DEG:
            self._log(f"[PULL_TIP] Tipping complete (pitch={pitch_deg:.1f}°). Returning to pre-squash pose.")
            self.tip_achieved = True
            self._enter_phase(Phase.RETURN_PRE_SQUASH)
            return

        # --- Slip detection ---
        ee_pos = self.irb.get_site_pose("ball")[:3, 3]
        if not self._pull_path_pos or np.linalg.norm(ee_pos - self._pull_path_pos[-1]) > 5e-4:
            self._pull_path_pos.append(ee_pos.copy())
        now    = self.data.time

        # Record pull start position and pitch on first active (non-stabilising) step
        if self._pull_start_x is None:
            self._pull_start_x = ee_pos[0]
        if self._pull_start_pitch is None:
            self._pull_start_pitch = pitch_deg

        total_travel       = abs(ee_pos[0] - self._pull_start_x)
        cumulative_pitch   = abs(pitch_deg - self._pull_start_pitch)

        self._slip_window_buf.append((now, ee_pos[0], pitch_deg))

        # Prune entries older than window
        while self._slip_window_buf and (now - self._slip_window_buf[0][0]) > self.SLIP_WINDOW:
            self._slip_window_buf.popleft()

        # Once the object has started responding (cumulative pitch > 0.1° from pull start),
        # disable slip detection — we're in tipping mode, not slip mode.
        tipping_started = cumulative_pitch > 0.1

        # Only evaluate slip after minimum lateral travel — avoids false positives
        # during the stabilisation window and initial contact transients.
        if not tipping_started and total_travel >= self.SLIP_MIN_TRAVEL and len(self._slip_window_buf) >= 2:
            t_old, x_old, p_old = self._slip_window_buf[0]
            delta_x_ee  = abs(ee_pos[0] - x_old)
            delta_pitch = abs(pitch_deg - p_old)

            if delta_x_ee > self.SLIP_EE_THRESH and delta_pitch < self.SLIP_PITCH_THRESH:
                self._log(f"[PULL_TIP] Slip detected! total_travel={total_travel*1000:.1f} mm, "
                      f"Δx_ee={delta_x_ee*1000:.1f} mm, Δpitch={delta_pitch:.2f}°, "
                      f"cumulative_pitch={cumulative_pitch:.3f}°")
                self._handle_slip()

    def _run_return_pre_squash(self):
        # Preferred return mode: invert pull-tip controller (same squash force loop,
        # same speed magnitude, opposite x direction) until we reach pull start.
        # The arc/path-replay implementation below is intentionally kept for future use.
        if self.RETURN_USE_REVERSE_PULL:
            self._run_return_pre_squash_reverse_pull()
            return

        """Replay pull path in reverse with arc lift and reduced normal force."""
        if not self._return_path:
            raw_path = self._pull_path_pos if self._pull_path_pos else [self.irb.get_site_pose("ball")[:3, 3].copy()]
            rev = list(reversed(raw_path[::max(1, int(self.RETURN_PATH_STRIDE))]))
            if raw_path:
                last = raw_path[0].copy()
                if np.linalg.norm(rev[-1] - last) > 1e-6:
                    rev.append(last)
            if self._pull_anchor_pos is not None:
                if np.linalg.norm(rev[-1] - self._pull_anchor_pos) > 1e-6:
                    rev.append(self._pull_anchor_pos.copy())

            n = len(rev)
            path = []
            for i, p in enumerate(rev):
                alpha = i / max(1, n - 1)
                lift = self.RETURN_ARC_Z * np.sin(np.pi * alpha)
                wp = p.copy()
                wp[2] += lift
                path.append(wp)

            self._return_path = path
            self._return_wp_idx = 0
            self._log(f"[RETURN_PRE_SQUASH] Replaying {len(self._return_path)} pull-path waypoints with arc lift.")

        if self._q_des is None:
            self._q_des = self.data.qpos[self.irb.joint_idx].copy().astype(float)
        if self._R_des is None:
            self._R_des = self.irb.FK()[:3, :3].copy()

        wp = self._return_path[self._return_wp_idx]
        ball_pos = self.irb.get_site_pose("ball")[:3, 3]
        diff = wp - ball_pos
        dist = np.linalg.norm(diff)
        v_lin = (diff / dist) * self.RETURN_PATH_SPEED if dist > 1e-6 else np.zeros(3)

        # Force-aware return: ramp normal-force target down along the replay,
        # with asymmetric z control (fast unload, gentle reload).
        alpha = self._return_wp_idx / max(1, len(self._return_path) - 1)
        f_z = abs(self._get_ft_world()[2])
        fz_start = self._return_fz_start if self._return_fz_start is not None else self.RETURN_FZ_START
        fz_target = (1.0 - alpha) * fz_start + alpha * self.RETURN_FZ_TARGET
        fz_err = fz_target - f_z
        vz_force = -self.RETURN_FZ_KP * fz_err
        if vz_force >= 0.0:
            vz_force = min(vz_force, self.RETURN_FZ_VMAX)
        else:
            vz_force = max(vz_force, -self.RETURN_FZ_DOWN_VMAX)
        v_lin[2] += vz_force

        # If contact weakens, slow lateral replay to recover grip before continuing.
        if f_z < self.RETURN_FORCE_RECOVER_THRESH * max(fz_target, 1e-6):
            v_lin[0] *= self.RETURN_FORCE_SLOW_RATIO
            v_lin[1] *= self.RETURN_FORCE_SLOW_RATIO

        R_curr = self.irb.FK()[:3, :3]
        R_err = self._R_des @ R_curr.T
        rotvec = R.from_matrix(R_err).as_rotvec()
        w_ori = self.ORI_KP * rotvec

        v_cmd = np.zeros(6)
        v_cmd[:3] = w_ori
        v_cmd[3:] = v_lin
        self._apply_cartesian_twist(v_cmd)

        if np.linalg.norm(ball_pos - wp) < self.RETURN_WP_TOL:
            self._return_wp_idx += 1
            if self._return_wp_idx >= len(self._return_path):
                self._log("[RETURN_PRE_SQUASH] Arc return complete. Done.")
                self._enter_phase(Phase.DONE)

    def _run_return_pre_squash_reverse_pull(self):
        """Return by inverting pull direction while maintaining squash force."""
        if self._q_des is None:
            self._q_des = self.data.qpos[self.irb.joint_idx].copy().astype(float)

        ee_pos = self.irb.get_site_pose("ball")[:3, 3]
        if self._pull_anchor_pos is not None and ee_pos[0] >= (self._pull_anchor_pos[0] - self.RETURN_X_TOL):
            self._log("[RETURN_PRE_SQUASH] Reverse-pull return reached pre-pull x. Done.")
            self._enter_phase(Phase.DONE)
            return

        ft = self._get_ft_world()
        f_z = abs(ft[2])
        dt = float(self.model.opt.timestep)

        fz_err = self._squash_force_target - f_z
        vz_cmd = -self.PULL_FZ_KP * fz_err
        vz_cmd = float(np.clip(vz_cmd, -self.PULL_Z_VEL_MAX, self.PULL_Z_VEL_MAX))

        v_cmd = np.zeros(6)
        v_cmd[3] = +self.PULL_SPEED   # reverse of pull phase
        v_cmd[5] = vz_cmd             # same squash-force regulation as pull
        self._apply_cartesian_twist(v_cmd)

    def _handle_slip(self):
        """Increment squash force and retry from RETREAT, or give up."""
        self._slip_retries += 1
        if self._slip_retries > self.MAX_SLIP_RETRIES:
            self._log(f"[PULL_TIP] Max slip retries ({self.MAX_SLIP_RETRIES}) reached. Returning to pre-squash pose.")
            self._enter_phase(Phase.RETURN_PRE_SQUASH)
            return

        new_f = min(self._squash_force_target * 1.5, self.F_SQUASH_MAX)
        self._log(f"[PULL_TIP] Retry {self._slip_retries}/{self.MAX_SLIP_RETRIES}. "
              f"Increasing squash force: {self._squash_force_target:.1f} → {new_f:.1f} N")
        self._squash_force_target = new_f
        self._squash_hold_start   = None
        self._pos_target          = None
        self._q_target            = None
        self._slip_lift           = True   # signal RETREAT to lift straight up, skip x-retract
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
        # Squash x: front edge inset by one and a half ball radii so the ball lands just inside the edge.
        # This maximises tipping leverage compared to squashing at the centroid.
        self.obj_squash_x   = float(aabb_min[0]) + 1.1 * self.irb.ball_radius # small inset for slip allowance

        if verbose:
            self._log(f"[SCAN] Object geometry: "
                  f"top_z={self.obj_top_z:.3f}, centroid_z={self.obj_centroid_z:.3f}, "
                  f"front_x={self.obj_front_x:.3f}, center_x={self.obj_center_x:.3f}, "
                  f"squash_x={self.obj_squash_x:.3f}")

    def _compute_retreat_waypoints(self, ee_pos: np.ndarray) -> list:
        """
        Return a list of (3,) positions describing the retreat path:
          1. Pull back in -x to clear the front face
          2. Rise to above the object top
          3. Advance to above the squash target (front edge) for maximum tipping leverage
        """
        clearance_x = self.obj_front_x - self.RETREAT_CLEARANCE
        above_z     = self.obj_top_z + self.TOP_CLEARANCE + self.irb.ball_radius

        wp1 = np.array([clearance_x,        0.0, ee_pos[2]])   # pull back at same height
        wp2 = np.array([clearance_x,        0.0, above_z])      # rise up
        wp3 = np.array([self.obj_squash_x,  0.0, above_z])      # move over front edge

        return [wp1, wp2, wp3]

    def _get_obj_pitch_deg(self) -> float:
        """Return signed pitch (deg) from quaternion y component.

        Assumes tipping is predominantly about world Y, so
        theta_y ~= 2*asin(q_y) with quaternion in [x, y, z, w].
        """
        q = np.asarray(self.irb.get_payload_pose(out='quat'), dtype=float).reshape(4)
        pitch_B     = R.from_quat(q).as_euler('xyz', degrees=True)[1]
        # qy = float(np.clip(q[1], -1.0, 1.0))
        # return float(np.degrees(2.0 * np.arcsin(qy)))
        return pitch_B

    def _get_ft_world(self) -> np.ndarray:
        """Return the F/T reading rotated into the world frame.

        ft_get_reading() returns forces in the sensor's local frame.  Rotating
        by R_sensor (sensor axes expressed in world) gives world-frame forces,
        so ft_world[2] is always the vertical (world-z) component regardless of
        wrist orientation.
        """
        ft_sensor = self.irb.ft_get_reading(flip_sign=True)
        R_sensor  = self.data.site_xmat[self.irb.ft_site].reshape(3, 3)
        f_world   = R_sensor @ ft_sensor[:3]
        t_world   = R_sensor @ ft_sensor[3:]
        return np.concatenate([f_world, t_world])

    # ------------------------------------------------------------------
    # Control helpers
    # ------------------------------------------------------------------

    def _hold_still(self):
        raise NotImplementedError("StateMachine subclasses must implement _hold_still().")

    def _apply_cartesian_twist(self, v_cmd: np.ndarray):
        raise NotImplementedError(
            "StateMachine subclasses must implement _apply_cartesian_twist()."
        )

    def _ensure_desired_orientation(self):
        if self._R_des is None:
            self._R_des = self.irb.FK()[:3, :3].copy()

    def _build_cartesian_twist(self, ball_target_pos: np.ndarray, speed: float) -> np.ndarray:
        self._ensure_desired_orientation()

        ball_pos = self.irb.get_site_pose("ball")[:3, 3]
        diff = ball_target_pos - ball_pos
        dist = np.linalg.norm(diff)

        v_lin = (diff / dist) * speed if dist > 1e-6 else np.zeros(3)

        R_curr = self.irb.FK()[:3, :3]
        R_err = self._R_des @ R_curr.T
        rotvec = R.from_matrix(R_err).as_rotvec()
        w_ori = self.ORI_KP * rotvec

        v_cmd = np.zeros(6)
        v_cmd[:3] = w_ori
        v_cmd[3:] = v_lin
        return v_cmd

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
        Build a Cartesian twist that steps the ball site toward ball_target_pos
        at `speed` m/s, then hand it off to the active actuator controller.
        """
        mujoco.mj_forward(self.model, self.data)
        v_cmd = self._build_cartesian_twist(ball_target_pos, speed)
        self._apply_cartesian_twist(v_cmd)

    # ------------------------------------------------------------------
    # Phase transition
    # ------------------------------------------------------------------

    def _enter_phase(self, new_phase: Phase):
        t_now = self.data.time
        if self.phase in self._phase_start_time:
            self._phase_end_time[self.phase] = t_now
        self._phase_start_time[new_phase] = t_now
        self._phase_settle_until = t_now + 0.1   # ignore force safety for 100 ms after transition
        if new_phase == Phase.SQUASH:
            if self._q_des is not None:
                self._q_pre_squash = self._q_des.copy()
            else:
                self._q_pre_squash = self.data.qpos[self.irb.joint_idx].copy().astype(float)
            self._squash_force_ready_since = None
            self._squash_fz_window.clear()
        if new_phase == Phase.PULL_TIP:
            self._pull_stable_until = t_now + 0.3  # hold z-only for 300 ms before lateral pull
            self._q_squash = self._q_des.copy() if self._q_des is not None else None
            self._pull_start_x = None     # will be set on first active pull step
            self._pull_start_pitch = None # pitch at pull start, for cumulative change gate
            self._pull_anchor_pos = self.irb.get_site_pose("ball")[:3, 3].copy()
            self._pull_path_pos = [self._pull_anchor_pos.copy()]
            self._return_path = []
            self._return_wp_idx = 0
        if new_phase == Phase.RETURN_PRE_SQUASH:
            self._return_fz_start = abs(self._get_ft_world()[2])
        self._log("")
        self._log(f"[PhaseController] {self.phase.name} → {new_phase.name}  (t={t_now:.3f} s)")
        self.phase = new_phase

        # Clear waypoint target so the new phase recomputes its own goals
        if new_phase in (Phase.APPROACH_PUSH, Phase.RETREAT):
            self._pos_target = None

        # Keep accumulated joint target across smooth motion chains:
        #   APPROACH_PUSH → PUSH  (continue forward)
        #   SQUASH → PULL_TIP     (preserve squash depth so force is not released)
        #   PULL_TIP → RETURN_PRE_SQUASH (smoothly interpolate back to pre-squash state)
        # All other transitions reset so the new phase starts from actual robot state.
        if new_phase not in (Phase.PUSH, Phase.PULL_TIP, Phase.RETURN_PRE_SQUASH):
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
