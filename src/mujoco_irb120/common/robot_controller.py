import mujoco
from scipy.spatial.transform import Rotation as Robj
import numpy as np
import matplotlib.pyplot as plt
from .helper_fns import *

class controller:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, ee_site='site:tool0'):
        self.model          = model
        self.data           = data
        self.joint_names    = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.joint_idx      = np.array([model.joint(name).qposadr for name in self.joint_names]).flatten() # qpos indices (for qpos, jnt_range, ctrl)
        self.joint_dof_idx  = np.array([model.joint(name).dofadr  for name in self.joint_names]).flatten() # velocity DOF indices (for Jacobian columns)
        self.ee_site        = model.site(ee_site).id
        self.table_site     = model.site('surface_site').id
        self.obj_frame_site = model.site('obj_frame_site').id
        self.o_obj          = self.data.site_xpos[self.obj_frame_site]  # (3,)
        self.payload_body_id = int(self.model.site_bodyid[self.obj_frame_site])
        self.pusher_body_id = int(model.body('pusher_link').id)
        try:
            self.pusher_geom_id = model.geom('push_rod').id
        except Exception:
            self.pusher_geom_id = None
        self.stop           = False                              # Stop flag for the controller
        self.q_min          = model.jnt_range[self.joint_idx, 0] # Max joint limits
        self.q_max          = model.jnt_range[self.joint_idx, 1] # Min joint limits
        self.v_max          = 1.5

        # --- Common Controller Variables ---
        self.J              = np.zeros((6, 6))                   # Size 6 x num_joints
        self.J_pinv         = np.zeros((6, 6))                   # Pseudo-inverse of the Jacobian
        self.T              = np.eye(4)                          # Current end-effector pose (4x4)
        self.R_desired      = np.eye(3)                          # Desired end-effector orientation (3x3)
        self.v_admittance   = np.zeros(6)                        # Stores the velocity for admittance control
        self.traj_coeffs    = np.zeros((6, 3))                   # Shape (6, 3) for each joint
        self.traj_duration  = 0.0                                # Duration of the trajectory
        self.traj_start_time = 0                                 # Start time of the trajectory

        # --- Manipulability Parameters ---
        self.a_margin       = 1.22 * 0.98                        # from mfg ellipsoid (2% margin) # 0.58, 0.87
        self.c_margin       = 1.74 * 0.98

        # --- Inverse Kinematics Parameters ---
        self.error_history  = []
        self.prev_error     = np.inf
        
        # --- Cartesian Keyboard Control Parameters ---
        self.kb_goal_pose = None  # Legacy IK goal pose (kept for compatibility)
        self.kb_q_des = None      # Persistent joint target for keyboard control

        # --- Force Sensor Calculations ---
        self.f_sensor_id    = model.sensor('force_sensor').id
        self.t_sensor_id    = model.sensor('torque_sensor').id
        try:
            self.ft_site = model.site('sensor_site').id
        except Exception:
            # Fallback: if sensor_site is absent, use tool flange site.
            self.ft_site = self.ee_site
        self.ft_offset      = np.zeros(6)  # Force-torque sensor offset for biasing
        self.grav_offset    = np.zeros(6)  # Gravity compensation offset
        self.grav_mass      = 0.0339              # Gravity compensation mass


    def FK(self):
        """Forward kinematics to get the current end-effector pose"""
        mujoco.mj_forward(self.model, self.data)
        # Assemble FK from Mujoco state info
        R_curr = self.data.site_xmat[self.ee_site].reshape(3, 3)
        p_curr = self.data.site_xpos[self.ee_site].reshape(3, 1)
        self.T[:3, :3] = R_curr
        self.T[:3, 3] = p_curr.flatten()
        return self.T

    def get_ee_position(self, copy=True):
        """Get end-effector position in world frame as a shape-(3,) vector."""
        p = self.data.site_xpos[self.ee_site]
        return p.copy() if copy else p

    @property
    def ee_position(self):
        """Convenience property for a copied end-effector world position."""
        return self.get_ee_position(copy=True)
    
    def get_jacobian(self, set_pinv=True):
        """Calculate the Jacobian matrix for the end-effector site"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)
        J_pos = jacp[:, self.joint_dof_idx]
        J_rot = jacr[:, self.joint_dof_idx]
        self.J = np.vstack([J_rot, J_pos]).squeeze()
        if set_pinv:
            self.J_pinv = np.linalg.pinv(self.J)  # Pseudo-inverse of the Jacobian
        return self.J

    def ft_get_reading(self, grav_comp=True):
        """Get wrench in WORLD frame from the F/T sensor.

        Args:
            grav_comp: subtract gravity load from force component.
        """

        # MuJoCo sensor id is not the same as the sensordata address; use sensor_adr.
        f_adr = int(self.model.sensor_adr[self.f_sensor_id])
        t_adr = int(self.model.sensor_adr[self.t_sensor_id])
        f_meas = np.asarray(self.data.sensordata[f_adr:f_adr + 3], dtype=float)
        t_meas = np.asarray(self.data.sensordata[t_adr:t_adr + 3], dtype=float)
        w_site = np.concatenate([f_meas, t_meas]) #- np.asarray(self.ft_offset, dtype=float).reshape(6)

        # Sensor-site rotation from MuJoCo site xmat.
        # Use transpose (inverse) for this wrench component mapping.
        R_wsensor = self.data.site_xmat[self.ft_site].reshape(3, 3).T
        # Static reference kept intentionally:
        # R_wsensor = np.array([[0, 0, -1],
        #                       [0, 1, 0],
        #                       [1, 0, 0]])  # static 90 flip about Y-axis
        
        w_site_out = w_site

        # Always return world-frame wrench.
        f_world = R_wsensor @ w_site_out[:3]
        t_world = R_wsensor @ w_site_out[3:]
        w = np.concatenate([f_world, t_world])
        return w
    
    def set_pose(self, q=np.zeros((6,1))):
        """Forcibly set the robot to a specific joint configuration by ignoring dynamics"""
        if not self.is_in_ellipsoid():
            print("Warning: Desired pose is outside the manipulability ellipsoid.")
            return
        self.data.qpos[self.joint_idx] = q
        self.data.qvel[self.joint_dof_idx] = 0.0
        self.data.ctrl[:] = np.asarray(q).flatten()  # sync actuator targets so position controllers don't spring back
        mujoco.mj_fwdPosition(self.model, self.data)
        self.error_history = []
        self.kb_q_des = self.data.qpos[self.joint_idx].copy().astype(float)
        self.kb_goal_pose = None
    
    def IK(self, T_des, method=2, max_iters=500, tol=1e-3, damping=0.1, step_size=0.5):
        """
        Inverse Kinematics to achieve desired end-effector pose T_des.
        This function is non-destructive and restores the original robot state after execution.

        Args:
            T_des:     4x4 homogeneous transformation matrix
            method:    1 for Newton-Raphson, 2 for Damped Least Squares, 3 for Gradient Descent
            max_iters: Maximum number of iterations to run
            tol:       Tolerance for convergence
            damping:   Damping factor for Damped Least Squares or Gradient Descent
            step_size: Step size for the update (used in Gradient Descent)
        
        Returns: 
            np.ndarray: Joint angles that achieve the desired pose
        """
        if T_des.shape != (4, 4):
            raise ValueError("T_des must be a 4x4 homogeneous transform matrix.")

        # --- Save the original simulation state ---
        q_original = self.data.qpos[self.joint_idx].copy()

        # Initialize the IK algo with current joint pos
        q = q_original.copy()
        prev_error = np.inf
        
        # Use a try...finally block to ensure we restore the original state
        try:
            for i in range(max_iters):
                self.data.qpos[self.joint_idx] = q              # Update position from previous iteration
                mujoco.mj_fwdPosition(self.model, self.data)    # Update forward kinematics
                self.FK()                                       # Get current end-effector pose

                # --- Compute error ---                         # AKA which T gets me from T_curr to T_des
                T_e = np.linalg.inv(self.T) @ T_des             # By definition, this is in the body frame
                xi_e = ht2screw(T_e)                            # Convert to twist form
                xi_e_space = twistbody2space(xi_e, self.T)      # Convert to space twist form because Jacobian given in space frame from Mujoco

                self.error_history.append(xi_e_space)           # Log the errors for plotting (optional)

                if np.linalg.norm(xi_e_space) < tol:
                    # print(f"\nIK converged in {i} iterations.")
                    return q                                    # Return successful solution

                ## --- Back track dynamic damping size ---
                if np.linalg.norm(xi_e_space) > prev_error:
                    damping *= 0.5
                # else:
                #     damping *= 1.5
                prev_error = np.linalg.norm(xi_e_space)

                # --- Compute Jacobian ---
                self.get_jacobian()  # Update the Jacobian matrix
                
                ## --- Choose update method ---
                if method == 2:     # Damped least squares (Levenberg-Marquardt)
                    J_update = self.J.T @ np.linalg.pinv((self.J @ self.J.T) + (damping**2 * np.eye(6))).real
                elif method == 1:   # Newton-Raphson
                    J_update = self.J_pinv.real
                    # J_update = np.linalg.pinv(self.J.T @ self.J) @ self.J.T # This is other form, but less stable
                elif method == 3:   # Gradient descent
                    J_update =  damping * self.J.T
                else:
                    raise ValueError("Invalid method. Choose 1, 2, or 3.")

                # --- Update joint angles ---
                delta_q = J_update @ xi_e_space
                q += delta_q.flatten()
                q = np.clip(q, self.q_min, self.q_max)          # Clamp to joint limits

            # If loop finishes without converging
            raise RuntimeError(f"IK did not converge within {max_iters} iterations. Final error: {np.linalg.norm(xi_e_space):.3f}")

        finally:
            ## --- Restore the original state ---
            self.data.qpos[self.joint_idx] = q_original
            mujoco.mj_fwdPosition(self.model, self.data)
    
    def set_pos_ctrl(self, q_desired, check_ellipsoid=True):
        """Apply position control to the robot"""
        if check_ellipsoid and not self.is_in_ellipsoid():              # Check manipulability
            return
        # self.data.ctrl[:] = q_desired.reshape(6,1)
        self.data.ctrl[:] = q_desired.flatten()
        mujoco.mj_forward(self.model, self.data)    # Update forward kinematics after control input

    def set_vel_ctrl(self, v_desired, Kp_ori=0, damping=1e-4):
        """Apply velocity control to the robot
            v_desired: 6D vector [lin_vel, ang_vel] in world frame
            damping: Damping factor for the control input
        """
        self.get_jacobian()
        if not self.is_in_ellipsoid():              # Check manipulability
            self.data.ctrl[:] = np.zeros(6)         # Stop motion
            return
        # dq = self.diff_IK(v_desired, Kp_ori=Kp_ori, damping=damping)       # Stop motion if outside ellipsoid
        q_dot = self.J_pinv @ v_desired
        self.data.ctrl[:] = q_dot.flatten()
        mujoco.mj_forward(self.model, self.data)    # Update forward kinematics after control input

    def apply_cartesian_keyboard_ctrl(self, v_cmd, dt=None, maintain_orientation=True, verbose=False):
        """Apply cartesian keyboard control to move the end-effector.
        
        Maintains a goal pose across frames and integrates velocity commands.
        Uses IK to compute joint angles toward the goal.
        
        Args:
            v_cmd (np.ndarray): 6D command [wx, wy, wz, vx, vy, vz] (rad/s, m/s)
            dt (float): Time step. If None, uses model.opt.timestep
            maintain_orientation (bool): If True, maintains current orientation
            verbose (bool): If True, prints debug info
        
        Returns:
            bool: True if control applied successfully, False otherwise
        """
        if dt is None:
            dt = self.model.opt.timestep

        # Expected command ordering is [wx, wy, wz, vx, vy, vz].
        if v_cmd is None:
            v_cmd = np.zeros(6)

        # Maintain orientation by zeroing angular velocity commands.
        if maintain_orientation:
            v_cmd = np.asarray(v_cmd, dtype=float).copy()
            v_cmd[:3] = 0.0

        # Initialize persistent desired joints and actively hold current pose.
        if self.kb_q_des is None:
            self.kb_q_des = self.data.qpos[self.joint_idx].copy().astype(float)
            self.set_pos_ctrl(self.kb_q_des, check_ellipsoid=False)
            if verbose:
                print("[KB CTRL] Initialized and holding current joint target")

        try:
            # Differential IK in one step: qdot = J^+ * v_cmd
            self.get_jacobian(set_pinv=True)
            q_dot = self.J_pinv @ v_cmd

            # Clip joint velocity to configured limit for stable keyboard behavior.
            q_dot = np.clip(q_dot, -self.v_max, self.v_max)

            # Integrate and clamp to joint limits.
            self.kb_q_des = self.kb_q_des + q_dot.flatten() * dt
            self.kb_q_des = np.clip(self.kb_q_des, self.q_min, self.q_max)

            # Send desired joint positions to MuJoCo position actuators.
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
        """Get the position of the table surface, NOT COM"""
        mujoco.mj_forward(self.model, self.data)  # Ensure Mujoco state is updated
        # Get global position of table surface CoM
        tab_global_pos = self.data.site_xpos[self.table_site].flatten()
        # account for height of table surface
        tab_dims = self.model.geom('table').size
        # Calculate the position of the surface (top plane)
        surface_pos = tab_global_pos + np.array([0, 0, tab_dims[2]]).flatten()
        return surface_pos

    def is_in_ellipsoid(self):
        """Check if the robot is within the manipulability ellipsoid"""
        self.FK()
        p = self.T[:3, 3].flatten()
        r2 = ((p[0]**2 + p[1]**2) / self.a_margin**2) + (p[2]**2 / self.c_margin**2) # normalized-ellipsoid coordinate
        if r2 > 1.0:
            print(f'Robot is outside the manipulability ellipsoid.')
            self.stop = True
            return False
        return True

    def get_payload_pose(self, site='payload_site', out='T', degrees=False, frame='world'):
        """Unified payload state accessor.

        Args:
            site: payload site name.
            output: one of {'T', 'R', 'p', 'rpy', 'quat'}.
            degrees: when True, angular outputs are returned in degrees.
            frame: 'world' (default) or 'body' for axis-angle/RPY conventions.
            reset_ref: for 'axis_angle', set current pose as new reference.
        """
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
        
        if out == "quat":
            return Robj.from_matrix(Rw).as_quat()  # (x, y, z, w) format

        if out == 'rpy':
            R_eval = np.eye(3) if frame == 'body' else Rw
            return Robj.from_matrix(R_eval).as_euler('xyz', degrees=degrees)

        raise ValueError("Output must be one of 'T', 'R', 'p', 'rpy', 'quat'.")

    def check_topple(self):
        payload_angle = self.get_payload_pose(out='rpy', degrees=True)
        if np.isclose(np.any(payload_angle == 90), True, atol=1e-2):
            self.stop = True

    def check_contact(self):
        """Return True only when the pusher is contacting the payload."""
        for contact in self.data.contact:
            g0, g1 = int(contact.geom[0]), int(contact.geom[1])
            b0, b1 = int(self.model.geom_bodyid[g0]), int(self.model.geom_bodyid[g1])

            # Primary path: detect contact between any geom on pusher body and any geom on payload body.
            pusher_in_contact = (b0 == self.pusher_body_id) or (b1 == self.pusher_body_id)
            payload_in_contact = (b0 == self.payload_body_id) or (b1 == self.payload_body_id)
            if pusher_in_contact and payload_in_contact:
                return True

            # Fast path: known pusher geom id.
            if self.pusher_geom_id is not None:
                if g0 != self.pusher_geom_id and g1 != self.pusher_geom_id:
                    continue
                other_gid = g1 if g0 == self.pusher_geom_id else g0
                if int(self.model.geom_bodyid[other_gid]) == self.payload_body_id:
                    return True
                continue

            # Fallback path: name-based pusher detection if geom id is unavailable.
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
            if 'pusher_link' in geom_names:         # If contact is between pusher and payload, skip it
                continue
            else:
                contact_verts.append(contact.pos)
        return np.array(contact_verts)
    
    def init_com_cone_from_edge(edge_verts):
        """
        A minimal cone representation:
        - apex is the entire edge (any point along it)
        - direction is +Z (up)
        - half-angle is ~90° (very wide)
        """
        return {
            "p1": edge_verts[0].copy(),     # (3,)
            "p2": edge_verts[1].copy(),     # (3,)
            "dir": np.array([0, 0, 1]),     # (3,) unit vector
            "half_angle": np.pi/2 - 1e-6,   # radians (almost 90°)
        }

# ======================================================================================================
# Unused fns
# ======================================================================================================
'''
    def bias_ft_reading(self): # NOTE: BROKEN, SO UNUSED
        print("Biasing F/T sensor...")
        force_offset = self.ft_get_reading()
        self.ft_offset = force_offset.copy()
        print(f"Force offset: {self.ft_offset.flatten()}")

    def generate_quintic_trajectory(self, q_start, q_end, duration):
        """
        Generates coefficients for a quintic polynomial trajectory.

        Args:
            q_start (np.ndarray): Starting joint configuration.
            q_end (np.ndarray): Ending joint configuration.
            duration (float): The total time for the trajectory.

        Returns:
            np.ndarray: A (6xN) matrix of coefficients, where N is the number of joints.
        """
        # Solve for the coefficients for each joint independently
        # c0, c1, c2 are 0 due to rest-to-rest boundary conditions
        q_start = np.asarray(q_start).flatten()
        q_end = np.asarray(q_end).flatten()
        
        c0 = q_start
        c1 = np.zeros_like(q_start)
        c2 = np.zeros_like(q_start)
        
        # System of equations for c3, c4, c5 from boundary conditions at time T
        A = np.array([
            [duration**3, duration**4, duration**5],
            [3*duration**2, 4*duration**3, 5*duration**4],
            [6*duration, 12*duration**2, 20*duration**3]])
        
        b = np.array([
            q_end - q_start,
            np.zeros_like(q_start),
            np.zeros_like(q_start)])
        
        # Solve A*x = b for x = [c3, c4, c5] for each joint
        c345 = np.linalg.solve(A, b)
        
        print(f"\nGenerated a {duration:.2f} sec trajectory to reach final pose.")
        self.traj_coeffs = np.vstack([c0, c1, c2, c345])  # Shape (6, 3) for each joint
        self.traj_duration = duration
        self.traj_start_time = self.data.time  # Start time of the trajectory
        
        return np.vstack([c0, c1, c2, c345])

    def evaluate_trajectory(self, t, order=1):
        """Evaluates the trajectory position at a given time t. Order determines pos or vel traj"""
        if t < 0: t = 0
        if t > self.traj_duration: t = self.traj_duration
        coeffs = self.traj_coeffs
        if order == 1:      # Position control
            return coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5
        elif order == 2:    # Velocity control
            return coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2 + 4*coeffs[4]*t**3 + 5*coeffs[5]*t**4

    def update_velocity_control(self, Kp_joint=5.0):
        """
        Follows a pre-planned quintic trajectory using joint velocity control.
        This uses a feedforward velocity command plus a feedback position correction.
        """
        if self.traj_start_time < 0:
            return # No active trajectory
        
        elapsed_time = self.data.time - self.traj_start_time

        # --- Get Desired state from Trajectory ---
        dq_desired = self.evaluate_trajectory_vel(elapsed_time).reshape(6,1) # Feedforward (ideal) velocity at this time

        # Feedback Term: corrective velocity based on position error
        q_desired = self.evaluate_trajectory_pos(elapsed_time).reshape(6,1)
        q_current = self.data.qpos[self.joint_idx].reshape(6,1)
        q_error = q_desired - q_current

        # --- Combine Final Velocity Command ---
        dq_command = dq_desired + Kp_joint * q_error

        # --- Apply Command to Actuators ---
        self.data.ctrl[self.joint_idx] = dq_command

    def diff_IK(self, v_des, Kp_ori=2, damping=1e-4):
        """
        Solve differential kinematics to achieve desired end-effector velocity
        v_des: 6D vector [lin_vel, ang_vel] in world frame
        damping: Damping factor for the control input
        Returns: joint velocities
        """
        # --- Get Current State ---
        self.FK()
        self.get_jacobian()

        # --- Calculate Orientation Error ---
        R_current = self.T[:3, :3]
        error_o_mat = self.R_desired @ R_current.T
        error_o_axis_angle = Robj.from_matrix(error_o_mat).as_rotvec()

        # --- Create Corrective Angular Velocity ---
        dv_ang_corrective = Kp_ori * error_o_axis_angle.reshape(3, 1)

        # --- Create Final Target Twist ---
        v_des[:3] += dv_ang_corrective.flatten()

        # --- Compute velocity (twist) error---
        v_error = v_des - self.J @ self.data.qvel[self.joint_dof_idx].reshape(-1,)
        # --- Damped Least Squares ---
        dv = self.J_pinv @ np.linalg.solve(self.J @ self.J_pinv + (damping * np.eye(6)), v_error)

        # --- Limit joint velocities ---
        # return np.clip(dv, -self.v_max, self.v_max).reshape(6,1)  # No need for this anymore, we set limits in model xml file
        return dv.reshape(6, 1)

    def update_admittance_control(self, f_target_linear, M=0.1, D=5.0, Kp_ori=25.0, Kv_ori=5.0):
        """
        Primary control function for stable pushing using admittance control.
        The robot acts as a virtual mass-spring-damper, moving in response to forces.

        Args:
            f_target_linear (np.ndarray): The 3D force vector the robot should try to apply.
            M (float): The virtual mass of the end-effector.
            D (float): The virtual damping of the end-effector.
        """
        # --- 1. Get Current State and Forces ---
        self.FK()
        J = self.get_jacobian()
        f_current_ee = self.get_pushing_force().flatten() # Force felt by the EE

        # --- 2. Admittance Law ---
        # Calculate the force error
        f_err = f_target_linear - f_current_ee
        
        # The core of admittance control: F_err = M*a + D*v
        # We solve for the desired acceleration: a = (F_err - D*v) / M
        # We only control admittance in the linear X direction (pushing direction)
        v_current_linear = (J @ self.data.qvel[self.joint_dof_idx])[3:]
        
        # Calculate desired acceleration only in the push direction (world X)
        a_admittance_x = (f_err[0] - D * v_current_linear[0]) / M
        
        # Integrate acceleration to get the next velocity command for the push direction
        self.v_admittance[3] += a_admittance_x * self.model.opt.timestep # v_next = v_prev + a*dt
        
        # --- 3. Orientation Holding ---
        # Use a standard PD controller to hold orientation
        R_current = self.T[:3, :3]
        err_o_mat = self.R_desired_orientation @ R_current.T
        err_o_axis_angle = Robj.from_matrix(err_o_mat).as_rotvec()
        v_current_angular = (J @ self.data.qvel[self.joint_dof_idx])[:3].flatten()
        
        # Calculate desired angular velocity to correct orientation
        self.v_admittance[:3] = Kp_ori * err_o_axis_angle - Kv_ori * v_current_angular
        
        # --- 4. Convert EE Velocity to Joint Torques ---
        # We now have a desired EE velocity (v_admittance). We need to command torques
        # to achieve it. This is a lower-level tracking problem.
        # We use another PD controller in joint space for this.
        Kp_joint = 10.0
        Kv_joint = 1.0
        
        # Calculate desired joint velocities
        dq_desired = (np.linalg.pinv(J) @ self.v_admittance).reshape(6,1)
        
        # Calculate joint error
        err_q = dq_desired - self.data.qvel[self.joint_dof_idx]

        # --- 5. Calculate and Apply Final Torques ---
        gravity_comp = self.data.qfrc_bias[self.joint_dof_idx]
        tau_command = Kp_joint * err_q - Kv_joint * self.data.qvel[self.joint_dof_idx] #+ gravity_comp

        self.data.ctrl[self.joint_idx] = tau_command


    def update_velocity_ff_fb_control(self, Kp_pos=10.0, Kp_ori=5.0):
        """
        Primary control function for accurate path following with stable contact.
        Uses velocity control with feedforward and feedback terms.
        Call once per simulation step.
        """
        if self.traj_start_time < 0:
            return  # No active trajectory
        
        # --- 1. Calculate Desired State from Trajectory ---
        elapsed_time = self.data.time - self.traj_start_time

        if elapsed_time > self.traj_duration:
            p_desired = self.T_end[:3, 3]   # Hold the final position
            v_desired_feedforward = np.zeros(3)
        else:
            p_start = self.T_start[:3, 3]   # Interpolate position for a straight line
            p_end = self.T_end[:3, 3]
            p_desired = p_start + (p_end - p_start) * (elapsed_time / self.traj_duration)
            v_desired_feedforward = (p_end - p_start) / self.traj_duration

        # Desired orientation is constant for this task TODO: Implement orientation control
        R_desired = self.T_start[:3, :3]

        # --- 2. Calculate Current State ---
        T_current = self.FK()
        p_current = T_current[:3, 3]
        R_current = T_current[:3, :3]

        # --- 3. Calculate Errors for Feedback ---
        err_p = p_desired - p_current       # Position error

        err_o_mat = R_desired @ R_current.T # Orientation error as rotation matrix
        err_o_axis_angle = Robj.from_matrix(err_o_mat).as_rotvec()  # Convert to axis-angle representation

        # --- 4. Velocity Control Law (Feedforward + Feedback) ---
        # Feedback term adds a corrective velocity based on position error
        v_feedback = Kp_pos * err_p
        v_command_linear = v_desired_feedforward + v_feedback

        # Orientation is controlled purely by feedback to maintain a constant orientation
        v_command_angular = Kp_ori * err_o_axis_angle

        # Combine into a 6D twist vector [angular; linear]
        v_command_full = np.hstack([v_command_angular, v_command_linear]).reshape(6, 1)

        # --- 5. Convert Task-Space Velocity to Joint Velocities ---
        dv = self.J_pinv @ v_command_full.reshape(6, 1) # Damped Least Squares
        dv = np.clip(dv, -1.5, 1.5).reshape(6, 1)       # Limit joint velocities

        # --- 6. Apply the Join Velocity Command ---
        self.data.ctrl[self.joint_idx] = dv

    def update_operational_space_control(self):
        """
        This is the primary control function for straight-line motion and stable contact.
        It calculates and applies the necessary torques to follow a Cartesian trajectory.
        Call this once per simulation step.
        """
        if self.traj_start_time < 0:
            return  # No active trajectory

        # --- 1. Calculate Desired State from Trajectory ---
        elapsed_time = self.data.time - self.traj_start_time
        
        if elapsed_time > self.traj_duration:
            # Hold the final position
            p_desired = self.T_end[:3, 3]
            v_desired = np.zeros(3)
        else:
            # Interpolate position for a straight line
            p_start = self.T_start[:3, 3]
            p_end = self.T_end[:3, 3]
            p_desired = p_start + (p_end - p_start) * (elapsed_time / self.traj_duration)
            v_desired = (p_end - p_start) / self.traj_duration

        # For this task, we only control position, not orientation.
        # Desired orientation is constant.
        R_desired = self.T_start[:3, :3]
        
        # --- 2. Calculate Current State ---
        T_current = self.FK()
        p_current = T_current[:3, 3]
        R_current = T_current[:3, :3]
        
        # Get current end-effector velocity (linear and angular)
        J = self.get_jacobian()
        v_current_full = J @ self.data.qvel[self.joint_dof_idx]
        v_current_angular = v_current_full[:3].squeeze()  # Extract angular velocity (first 3 elements)
        v_current_linear = v_current_full[3:].squeeze()  # Extract linear velocity (last 3 elements)

        # --- 3. Calculate Errors ---
        # Position error
        err_p = p_desired - p_current
        # Velocity error
        err_v = v_desired - v_current_linear
        # Orientation error (simplified: axis-angle between desired and current)
        err_o = R_current.T @ R_desired
        err_o_axis_angle = Robj.from_matrix(err_o).as_rotvec()
        
        # Angular velocity error (desired is 0)
        err_o_vel = -v_current_angular # <-- Added this

        # --- 4. Define Gains for the OSC Controller ---
        # These gains relate to the "stiffness" and "damping" of the end-effector itself.
        kp_pos = 40.0  # Proportional gain for position
        kv_pos = 40.0   # Derivative gain for position (damping)
        kp_ori = 10.0  # Proportional gain for orientation
        kv_ori = 1.0   # Derivative gain for orientation (damping)

        # --- 5. The Operational Space Control Law ---
        # Calculate the desired force and torque to apply at the end-effector
        # to correct the errors.
        force_desired = (kp_pos * err_p) + (kv_pos * err_v)
        torque_desired = (kp_ori * err_o_axis_angle) + (kv_ori * err_o_vel)

        # Combine into a 6D wrench vector [torque; force]
        wrench_desired = np.hstack([torque_desired, force_desired]).reshape(6, 1)

        # --- 6. Map Wrench to Joint Torques ---
        # Use the Jacobian transpose to find the joint torques that produce the desired wrench.
        # Also, add gravity compensation to counteract the robot's own weight.
        gravity_compensation = self.data.qfrc_bias[self.joint_idx].reshape(6,1)
        tau_command = J.T @ wrench_desired #+ gravity_compensation
        
        # --- 7. Apply the Torques ---
        # This assumes your actuators are of type <motor> in the XML.
        self.data.ctrl[self.joint_idx] = tau_command
'''