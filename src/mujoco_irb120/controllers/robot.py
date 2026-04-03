import mujoco
from scipy.spatial.transform import Rotation as Robj
import numpy as np
from mujoco_irb120.util.helper_fns import *


class Robot:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model          = model
        self.data           = data
        self.joint_names    = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.joint_idx      = np.array([model.joint(name).qposadr for name in self.joint_names]).flatten()
        self.joint_dof_idx  = np.array([model.joint(name).dofadr  for name in self.joint_names]).flatten()
        self.stop           = False
        self.q_min          = model.jnt_range[self.joint_idx, 0]
        self.q_max          = model.jnt_range[self.joint_idx, 1]
        self.v_max          = 1.5

        self.J              = np.zeros((6, 6))
        self.J_pinv         = np.zeros((6, 6))
        self.T              = np.eye(4)
        self.R_desired      = np.eye(3)
        self.v_admittance   = np.zeros(6)
        self.traj_coeffs    = np.zeros((6, 3))
        self.traj_duration  = 0.0
        self.traj_start_time = 0

        self.a_margin       = 1.22 * 0.98
        self.c_margin       = 1.74 * 0.98

        self.error_history  = []
        self.prev_error     = np.inf

        self.kb_goal_pose = None
        self.kb_q_des = None

        self.ee_site        = model.site('site:tool0').id
        self.table_site     = model.site('site:table').id
        self.fingertip_site = model.site('site:fingertip').id
        self.ball_site      = model.site('site:ball_center').id
        self.obj_frame_site = model.site('site:obj_frame').id
        self.payload_body_id = int(self.model.site_bodyid[self.obj_frame_site])
        self.pusher_body_id = int(model.body('pusher_link').id)
        self.ball_geom_id   = model.geom('robot0:push_ball_col').id

        self.f_adr          = int(self.model.sensor_adr[model.sensor('force_sensor').id])
        self.t_adr          = int(self.model.sensor_adr[model.sensor('torque_sensor').id])
        self.ft_site        = model.site('site:sensor').id
        self.ft_bias_val    = np.zeros(6)
        self.grav_mass      = float(np.asarray(model.body('pusher_link').mass).flat[0])
        self.ball_radius    = float(model.geom_size[self.ball_geom_id, 0])

    def FK(self):
        mujoco.mj_forward(self.model, self.data)
        R_curr = self.data.site_xmat[self.ee_site].reshape(3, 3)
        p_curr = self.data.site_xpos[self.ee_site].reshape(3, 1)
        self.T[:3, :3] = R_curr
        self.T[:3, 3] = p_curr.flatten()
        return self.T

    def get_site_pose(self, which='ee'):
        site_map = {
            'ee':     self.ee_site,
            'ball':   self.ball_site,
            'sensor': self.ft_site,
        }
        if which not in site_map:
            raise ValueError(f"which must be one of {set(site_map)}")
        sid = site_map[which]
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[sid].reshape(3, 3).copy()
        T[:3,  3] = self.data.site_xpos[sid].copy()
        return T

    def get_jacobian(self, set_pinv=True):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)
        J_pos = jacp[:, self.joint_dof_idx]
        J_rot = jacr[:, self.joint_dof_idx]
        self.J = np.vstack([J_rot, J_pos]).squeeze()
        if set_pinv:
            self.J_pinv = np.linalg.pinv(self.J)
        return self.J

    def ft_bias(self, n_samples=200):
        print(f"Biasing F/T sensor with {n_samples} static samples...")
        samples = []

        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        for _ in range(n_samples):
            mujoco.mj_step(self.model, self.data)
            samples.append(self.ft_get_reading(grav_comp=True, apply_bias=False))

        self.ft_bias_val = np.mean(np.asarray(samples, dtype=float), axis=0)
        print(f"Force offset: {self.ft_bias_val}")
        return self.ft_bias_val.copy()

    def ft_get_reading(self, grav_comp=True, apply_bias=True, flip_sign=True):
        f_S = np.asarray(self.data.sensordata[self.f_adr:self.f_adr + 3], dtype=float).copy()
        t_S = np.asarray(self.data.sensordata[self.t_adr:self.t_adr + 3], dtype=float).copy()

        if grav_comp:
            R_BS = self.data.site_xmat[self.ft_site].reshape(3, 3)
            g_S = R_BS.T @ np.asarray(self.model.opt.gravity, dtype=float)
            weight_S = self.grav_mass * g_S
            f_S -= weight_S
            r_B = self.data.site_xpos[self.fingertip_site] - self.data.site_xpos[self.ft_site]
            r_S = R_BS.T @ r_B
            t_S -= np.cross(r_S, weight_S)

        w = np.concatenate([f_S, t_S])
        w *= -1 if flip_sign else 1
        return w - self.ft_bias_val if apply_bias else w

    def set_pose(self, q=np.zeros((6, 1))):
        if not self.is_in_ellipsoid():
            print("Warning: Desired pose is outside the manipulability ellipsoid.")
            return
        self.data.qpos[self.joint_idx] = q
        self.data.qvel[self.joint_dof_idx] = 0.0
        self.set_pos_ctrl(np.asarray(q).flatten(), check_ellipsoid=False)
        mujoco.mj_fwdPosition(self.model, self.data)
        self.error_history = []
        self.kb_q_des = self.data.qpos[self.joint_idx].copy().astype(float)
        self.kb_goal_pose = None

    def IK(self, T_des, method=2, max_iters=500, tol=1e-3, damping=0.1, step_size=0.5):
        if T_des.shape != (4, 4):
            raise ValueError("T_des must be a 4x4 homogeneous transform matrix.")

        q_original = self.data.qpos[self.joint_idx].copy()
        q = q_original.copy()
        prev_error = np.inf

        try:
            for _ in range(max_iters):
                self.data.qpos[self.joint_idx] = q
                mujoco.mj_fwdPosition(self.model, self.data)
                self.FK()

                T_e = np.linalg.inv(self.T) @ T_des
                xi_e = ht2screw(T_e)
                xi_e_space = twistbody2space(xi_e, self.T)
                self.error_history.append(xi_e_space)

                if np.linalg.norm(xi_e_space) < tol:
                    return q

                if np.linalg.norm(xi_e_space) > prev_error:
                    damping *= 0.5
                prev_error = np.linalg.norm(xi_e_space)

                self.get_jacobian()

                if method == 2:
                    J_update = self.J.T @ np.linalg.pinv((self.J @ self.J.T) + (damping**2 * np.eye(6))).real
                elif method == 1:
                    J_update = self.J_pinv.real
                elif method == 3:
                    J_update = damping * self.J.T
                else:
                    raise ValueError("Invalid method. Choose 1, 2, or 3.")

                delta_q = J_update @ xi_e_space
                q += delta_q.flatten()
                q = np.clip(q, self.q_min, self.q_max)

            raise RuntimeError(f"IK did not converge within {max_iters} iterations. Final error: {np.linalg.norm(xi_e_space):.3f}")

        finally:
            self.data.qpos[self.joint_idx] = q_original
            mujoco.mj_fwdPosition(self.model, self.data)

    def set_pos_ctrl(self, q_desired, check_ellipsoid=True):
        raise NotImplementedError

    def set_vel_ctrl(self, v_desired, Kp_ori=0, damping=1e-4):
        raise NotImplementedError

    def apply_cartesian_keyboard_ctrl(self, v_cmd, dt=None, maintain_orientation=True, verbose=False):
        if dt is None:
            dt = self.model.opt.timestep

        if v_cmd is None:
            v_cmd = np.zeros(6)

        if maintain_orientation:
            v_cmd = np.asarray(v_cmd, dtype=float).copy()
            v_cmd[:3] = 0.0

        if self.kb_q_des is None:
            self.kb_q_des = self.data.qpos[self.joint_idx].copy().astype(float)
            self.set_pos_ctrl(self.kb_q_des, check_ellipsoid=False)
            if verbose:
                print("[KB CTRL] Initialized and holding current joint target")

        try:
            self.get_jacobian(set_pinv=True)
            q_dot = self.J_pinv @ v_cmd
            q_dot = np.clip(q_dot, -self.v_max, self.v_max)
            self.kb_q_des = self.kb_q_des + q_dot.flatten() * dt
            self.kb_q_des = np.clip(self.kb_q_des, self.q_min, self.q_max)
            self.set_pos_ctrl(self.kb_q_des, check_ellipsoid=False)

            if verbose and not np.allclose(v_cmd, 0):
                ee = self.FK()[:3, 3]
                print(f"[KB CTRL] cmd v=({v_cmd[3]:+.3f}, {v_cmd[4]:+.3f}, {v_cmd[5]:+.3f}) m/s | EE=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")
            return True

        except Exception as e:
            if verbose:
                print(f"[KB CTRL] Error: {str(e)[:80]}")
            return False

    def get_surface_pos(self):
        mujoco.mj_forward(self.model, self.data)
        tab_global_pos = self.data.site_xpos[self.table_site].flatten()
        tab_dims = self.model.geom('table').size
        surface_pos = tab_global_pos + np.array([0, 0, tab_dims[2]]).flatten()
        return surface_pos

    def is_in_ellipsoid(self):
        self.FK()
        p = self.T[:3, 3].flatten()
        r2 = ((p[0]**2 + p[1]**2) / self.a_margin**2) + (p[2]**2 / self.c_margin**2)
        if r2 > 1.0:
            print('Robot is outside the manipulability ellipsoid.')
            self.stop = True
            return False
        return True

    def get_payload_pose(self, site='site:obj_frame', out='T', degrees=False, frame='world'):
        sid = self.model.site(site).id
        Rw = self.data.site_xmat[sid].reshape(3, 3)
        p = self.data.site_xpos[sid].flatten()

        if out == 'T':
            T_payload = np.eye(4)
            T_payload[:3, :3] = Rw
            T_payload[:3, 3] = p
            return T_payload

        if out == 'R':
            return Rw

        if out == 'p':
            return p

        if out == 'quat':
            return Robj.from_matrix(Rw).as_quat()

        if out == 'rpy':
            R_eval = np.eye(3) if frame == 'body' else Rw
            return Robj.from_matrix(R_eval).as_euler('xyz', degrees=degrees)

        raise ValueError("Output must be one of 'T', 'R', 'p', 'rpy', 'quat'.")

    def check_topple(self):
        payload_angle = self.get_payload_pose(out='rpy', degrees=True)
        if np.isclose(np.any(payload_angle == 90), True, atol=1e-2):
            self.stop = True

    def check_contact(self):
        for contact in self.data.contact:
            g0, g1 = int(contact.geom[0]), int(contact.geom[1])
            b0, b1 = int(self.model.geom_bodyid[g0]), int(self.model.geom_bodyid[g1])

            pusher_in_contact = (b0 == self.pusher_body_id) or (b1 == self.pusher_body_id)
            payload_in_contact = (b0 == self.payload_body_id) or (b1 == self.payload_body_id)
            if pusher_in_contact and payload_in_contact:
                return True

            if self.ball_geom_id is not None:
                if g0 != self.ball_geom_id and g1 != self.ball_geom_id:
                    continue
                other_gid = g1 if g0 == self.ball_geom_id else g0
                if int(self.model.geom_bodyid[other_gid]) == self.payload_body_id:
                    return True
                continue

            names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or '' for gid in (g0, g1)]
            pusher_flags = [('push_rod' in n) or ('pusher_link' in n) for n in names]
            if not any(pusher_flags):
                continue
            other_gid = g1 if pusher_flags[0] else g0
            if int(self.model.geom_bodyid[other_gid]) == self.payload_body_id:
                return True
        return False

    def get_tip_edge(self):
        contact_verts = []
        for contact in self.data.contact:
            geom_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(id)) for id in contact.geom]
            if 'pusher_link' in geom_names:
                continue
            contact_verts.append(contact.pos)
        return np.array(contact_verts)

    @staticmethod
    def init_com_cone_from_edge(edge_verts):
        return {
            "p1": edge_verts[0].copy(),
            "p2": edge_verts[1].copy(),
            "dir": np.array([0, 0, 1]),
            "half_angle": np.pi/2 - 1e-6,
        }