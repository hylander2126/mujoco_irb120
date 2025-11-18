import mujoco
from scipy.spatial.transform import Rotation as Robj
import numpy as np
import matplotlib.pyplot as plt
from utils.helper_fns import *

class controller:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, ee_site='site:tool0'):
        self.model          = model
        self.data           = data
        self.joint_names    = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.joint_idx      = np.array([model.joint(name).qposadr for name in self.joint_names]) # This is same as dofadr (v_indices)
        self.ee_site        = model.site(ee_site).id
        self.table_site     = model.site('surface_site').id
        self.obj_frame_site = model.site('obj_frame_site').id
        self.o_obj          = self.data.site_xpos[self.obj_frame_site]  # (3,)
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

        # --- Force Sensor Calculations ---
        self.f_sensor_id    = model.sensor('force_sensor').id
        self.ft_offset      = np.zeros((3, 1))  # Force-torque sensor offset for biasing
        self.grav_offset    = np.zeros((3, 1))  # Gravity compensation offset
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
    
    def get_jacobian(self):
        """Calculate the Jacobian matrix for the end-effector site"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)
        J_pos = jacp[:, self.joint_idx]
        J_rot = jacr[:, self.joint_idx]
        self.J = np.vstack([J_rot, J_pos]).squeeze()
        self.J_pinv = np.linalg.pinv(self.J)  # Pseudo-inverse of the Jacobian
        return self.J

    def ft_get_reading(self, grav_comp=True):
        """Get the force reading from the force sensor, optionally gravity compensated."""
        f_meas = self.data.sensordata[self.f_sensor_id:self.f_sensor_id + 3].reshape(3, 1)

        # Apply any prior bias
        f = f_meas - self.ft_offset

        if not grav_comp:
            return f
        # Optional gravity comp (force only; torque left untouched since you read 3D force)
        # site rotation: site frame -> world; take transpose for world -> site
        R_sw = self.data.site_xmat[self.ee_site].reshape(3, 3).T
        # gravity force expressed in site frame
        f_g_site = R_sw @ (self.grav_mass * np.array([0, 0, -9.81])).reshape(3, 1)
        
        # f_g_site = np.zeros((3,1))  # DISABLE GRAVITY COMPENSATION FOR NOW

        return f - f_g_site  # subtract mg
    
    def set_pose(self, q=np.zeros((6,1))):
        """Forcibly set the robot to a specific joint configuration by ignoring dynamics"""
        if not self.is_in_ellipsoid():
            print("Warning: Desired pose is outside the manipulability ellipsoid.")
            return
        self.data.qpos[self.joint_idx] = q
        self.data.qvel[self.joint_idx] = 0.0
        mujoco.mj_fwdPosition(self.model, self.data)
        self.error_history = []
    
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
        
        # Use a try...finally block to ensure we restore the original state
        try:
            for i in range(max_iters):
                self.data.qpos[self.joint_idx] = q              # Update position from previous iteration
                mujoco.mj_fwdPosition(self.model, self.data)    # Update forward kinematics
                self.FK()                                       # Get current end-effector pose

                # --- Compute error ---                         # AKA which T gets me from T_curr to T_des
                T_e = np.linalg.inv(self.T) @ T_des        # By definition, this is in the body frame
                xi_e = ht2screw(T_e)                            # Convert to twist form
                xi_e_space = twistbody2space(xi_e, self.T) # Convert to space twist form because Jacobian given in space frame from Mujoco

                self.error_history.append(xi_e_space)           # Log the errors for plotting (optional)

                if np.linalg.norm(xi_e_space) < tol:
                    # print(f"\nIK converged in {i} iterations.")
                    return q                                    # Return successful solution
                
                # --- Compute Jacobian ---
                self.get_jacobian()  # Update the Jacobian matrix

                ## --- Back track dynamic damping size ---
                if np.linalg.norm(xi_e_space) > self.prev_error:
                    damping *= 0.5
                # else:
                #     damping *= 1.5
                self.prev_error = np.linalg.norm(xi_e_space)

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
                q += delta_q.reshape(6,1)
                q = np.clip(q, self.q_min, self.q_max)          # Clamp to joint limits

            # If loop finishes without converging
            print(f"\nIK did not converge within {max_iters} iterations. Final error norm: {np.linalg.norm(xi_e_space):.3f}")
            print("**********************************\n")
            return None

        finally:
            ## --- Restore the original state ---
            self.data.qpos[self.joint_idx] = q_original
            mujoco.mj_fwdPosition(self.model, self.data)
            print("\nIK finished, robot state restored.")
            print("**********************************")
    
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

    def get_payload_pose(self, site='payload_site', output='T', degrees=False):
        payload_site_id = self.model.site(site).id
        payload_R = self.data.site_xmat[payload_site_id].reshape(3, 3)
        payload_p = self.data.site_xpos[payload_site_id].flatten()
        T_payload = np.eye(4)
        T_payload[:3, :3] = payload_R
        T_payload[:3, 3] = payload_p
        if output == 'T':
            return T_payload
        elif output == 'pitch':
            rot = Robj.from_matrix(payload_R).as_euler('xyz', degrees=False)
            tip_angle = rot[1]  # pitch angle
            if degrees:
                tip_angle = np.degrees(tip_angle)
            return tip_angle
        elif output == 'p':
            return payload_p
        else:
            raise ValueError("Output must be 'pitch', 'p', or 'T'.")
        
    def get_payload_rpy(self, site='payload_site', degrees=False, frame='world'):
        """
        Get the Euler angles representation of the payload's orientation.
        """

        sid = self.model.site(site).id
        Rw = self.data.site_xmat[sid].reshape(3, 3)

        if frame == 'body':
            Rb = Rw.T @ Rw  # Identity, since Rw.T @ Rw = I
            R = Rb
        else:
            R = Rw

        rot = Robj.from_matrix(R)
        rpy = rot.as_euler('xyz', degrees=degrees)  # roll, pitch, yaw
        return rpy

    def get_payload_axis_angle(self, site='payload_site', degrees=False, frame='world', reset_ref=False):
        """
        Finite axis–angle of the payload's rotation relative to its initial pose.

        Args:
            site:       site name for payload
            degrees:    if True, return angle in degrees (else radians)
            frame:      'world' (default) or 'body' to express the axis
            reset_ref:  if True, set current pose as new zero-rotation reference

        Returns:
            angle, axis
            - angle: scalar >= 0 (rad or deg)
            - axis:  (3,) unit vector (undefined if angle≈0; we return [1,0,0])
        """
        sid = self.model.site(site).id
        Rw = self.data.site_xmat[sid].reshape(3, 3)

        # cache/refresh reference orientation
        if reset_ref or not hasattr(self, '_payload_R0'):
            self._payload_R0 = Rw.copy()
        R0 = self._payload_R0

        # relative rotation
        R_rel = Rw @ R0.T

        # axis–angle via rotation vector
        rotvec = Robj.from_matrix(R_rel).as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        axis = (rotvec / angle) if angle > 1e-12 else np.array([1.0, 0.0, 0.0])

        # express axis in desired frame
        if frame == 'body':
            axis = Rw.T @ axis

        if degrees:
            angle = np.rad2deg(angle)

        return angle, axis
        
    def check_topple(self):
        payload_angle = self.get_payload_pose(output='pitch', degrees=True)
        if np.isclose(payload_angle, 90, atol=1e-2):
            self.stop = True

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
        v_error = v_des - self.J @ self.data.qvel[self.joint_idx].reshape(-1,)
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
        v_current_linear = (J @ self.data.qvel[self.joint_idx])[3:]
        
        # Calculate desired acceleration only in the push direction (world X)
        a_admittance_x = (f_err[0] - D * v_current_linear[0]) / M
        
        # Integrate acceleration to get the next velocity command for the push direction
        self.v_admittance[3] += a_admittance_x * self.model.opt.timestep # v_next = v_prev + a*dt
        
        # --- 3. Orientation Holding ---
        # Use a standard PD controller to hold orientation
        R_current = self.T[:3, :3]
        err_o_mat = self.R_desired_orientation @ R_current.T
        err_o_axis_angle = Robj.from_matrix(err_o_mat).as_rotvec()
        v_current_angular = (J @ self.data.qvel[self.joint_idx])[:3].flatten()
        
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
        err_q = dq_desired - self.data.qvel[self.joint_idx]

        # --- 5. Calculate and Apply Final Torques ---
        gravity_comp = self.data.qfrc_bias[self.joint_idx]
        tau_command = Kp_joint * err_q - Kv_joint * self.data.qvel[self.joint_idx] #+ gravity_comp

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
        v_current_full = J @ self.data.qvel[self.joint_idx]
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