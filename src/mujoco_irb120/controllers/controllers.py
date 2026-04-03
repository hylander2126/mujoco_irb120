import mujoco
import numpy as np

from mujoco_irb120.controllers.robot import Robot


class PositionController(Robot):
    def set_pos_ctrl(self, q_desired, check_ellipsoid=True):
        if check_ellipsoid and not self.is_in_ellipsoid():
            return
        self.data.ctrl[:] = np.asarray(q_desired).flatten()
        mujoco.mj_forward(self.model, self.data)

    def set_vel_ctrl(self, v_desired, Kp_ori=0, damping=1e-4):
        self.get_jacobian()
        if not self.is_in_ellipsoid():
            self.set_pos_ctrl(self.data.qpos[self.joint_idx].copy(), check_ellipsoid=False)
            return
        q_dot = self.J_pinv @ np.asarray(v_desired, dtype=float).reshape(6)
        q_dot = np.clip(q_dot, -self.v_max, self.v_max)
        q_des = self.data.qpos[self.joint_idx].copy().astype(float) + q_dot.flatten() * self.model.opt.timestep
        self.set_pos_ctrl(q_des, check_ellipsoid=False)


class VelocityController(Robot):
    POS_TRACK_KP = 20.0

    def set_pos_ctrl(self, q_desired, check_ellipsoid=True):
        if check_ellipsoid and not self.is_in_ellipsoid():
            return
        q_current = self.data.qpos[self.joint_idx].copy().astype(float)
        q_error = np.asarray(q_desired, dtype=float).flatten() - q_current
        q_dot = np.clip(self.POS_TRACK_KP * q_error, -self.v_max, self.v_max)
        self.data.ctrl[:] = q_dot.flatten()
        mujoco.mj_forward(self.model, self.data)

    def set_vel_ctrl(self, v_desired, Kp_ori=0, damping=1e-4):
        self.get_jacobian()
        if not self.is_in_ellipsoid():
            self.data.ctrl[:] = np.zeros(6)
            return
        q_dot = self.J_pinv @ np.asarray(v_desired, dtype=float).reshape(6)
        q_dot = np.clip(q_dot, -self.v_max, self.v_max)
        self.data.ctrl[:] = q_dot.flatten()
        mujoco.mj_forward(self.model, self.data)