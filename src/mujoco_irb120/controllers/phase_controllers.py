"""Actuator-specific phase controller implementations.

The shared phase ordering, geometry helpers, and logging live in
state_machine.py. This module provides the concrete motion commands for
position and velocity actuators.
"""

import numpy as np

from mujoco_irb120.controllers.state_machine import StateMachine as PhaseStateMachine
from mujoco_irb120.controllers.state_machine import Phase


class PositionPhaseController(PhaseStateMachine):
    """Phase controller that drives the robot through position targets."""

    def _hold_still(self):
        if self._q_des is None:
            self._q_des = self.data.qpos[self.irb.joint_idx].copy().astype(float)
        self.irb.set_pos_ctrl(self._q_des, check_ellipsoid=False)

    def _apply_cartesian_twist(self, v_cmd: np.ndarray):
        if self._q_des is None:
            self._q_des = self.data.qpos[self.irb.joint_idx].copy().astype(float)

        dt = float(self.model.opt.timestep)
        self.irb.get_jacobian(set_pinv=True)
        q_dot = self.irb.J_pinv @ np.asarray(v_cmd, dtype=float).reshape(6)
        q_dot = np.clip(q_dot, -self.irb.v_max, self.irb.v_max)

        self._q_des = self._q_des + q_dot * dt
        self._q_des = np.clip(self._q_des, self.irb.q_min, self.irb.q_max)
        self.irb.set_pos_ctrl(self._q_des, check_ellipsoid=False)


class VelocityPhaseController(PhaseStateMachine):
    """Phase controller that drives the robot through velocity commands."""

    # Velocity control is more immediate, so keep the motions a bit gentler.
    MOVE_SPEED = 0.03
    PUSH_SPEED_CTRL = 0.02
    DESCEND_SPEED_CTRL = 0.015
    SQUASH_SPEED_MAX = 0.005 # 0.0025
    PULL_SPEED = 0.032
    RETURN_PATH_SPEED = 0.012

    def _hold_still(self):
        self.irb.set_vel_ctrl(np.zeros(6), damping=1e-4)

    def _apply_cartesian_twist(self, v_cmd: np.ndarray):
        self.irb.set_vel_ctrl(np.asarray(v_cmd, dtype=float).reshape(6))
