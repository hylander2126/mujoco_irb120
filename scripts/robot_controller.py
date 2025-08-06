import mujoco
from scipy.spatial.transform import Rotation as Robj
import numpy as np
import matplotlib.pyplot as plt
from helper_fns import *

class controller:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, ee_site='ee_site'):
        self.model = model
        self.data = data
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.joint_idx = np.array([model.joint(name).qposadr for name in self.joint_names]) # This is same as dofadr (v_indices)
        self.ee_site = model.site(ee_site).id
        self.table_site = model.site('surface_site').id
        # Initialize Jacobian matrices
        self.J = np.zeros((6, 6)) # Jacobian is size 6 x num_joints
        # Current end-effector pose
        self.T = np.eye(4)
        # Get joint limits
        self.q_min = model.jnt_range[self.joint_idx, 0]
        self.q_max = model.jnt_range[self.joint_idx, 1]
        # Set up manipulability ellipsoid parameters
        self.a_margin = 1.22 * 0.98 # from mfg ellipsoid (2% margin) # 0.58, 0.87
        self.c_margin = 1.74 * 0.98
        # For IK
        self.error_history = []
        self.prev_error = np.inf
        # For contact force calculation
        self.payload_geom_id = model.geom('payload').id
        self.table_geom_id = model.geom('table').id
        self.pusher_geom_id = model.geom('push_rod').id
        # Stop flag for the controller
        self.stop = False

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
        return self.J
    
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
    
    def set_velocity_control(self, v_desired, damping=1e-4):
        """Apply velocity control to the robot"""
        # ## FOR VELOCITY CONTROL (format: [vx vy vz wx wy wz])  Put this into the main file when desired
        # target_vel = np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0])

        # Check manipulability
        if not self.is_in_ellipsoid(): 
            self.data.ctrl[:] = np.zeros(6) # Stop motion
            # mujoco.mj_forward(self.model, self.data) # Update forward kinematics after control input
            return
        
        dq = self.diff_IK(v_desired, damping) # Stop motion if outside ellipsoid
        self.data.ctrl[:] = dq.flatten()
        mujoco.mj_forward(self.model, self.data) # Update forward kinematics after control input
    
    def set_pose(self, q=np.zeros((6,1))):
        """Reset robot to desired position, by default home position"""
        # Check Manipulability
        if not self.is_in_ellipsoid():
            print("Warning: Desired pose is outside the manipulability ellipsoid.")
            return
        self.data.qpos[self.joint_idx] = q
        self.data.qvel[self.joint_idx] = 0.0
        mujoco.mj_fwdPosition(self.model, self.data)
        self.error_history = []

    def plot_error(self, tol):
        """Plot error norm history with horizontal line at zero"""
        plt.figure(figsize=(12, 6))
        plt.plot(np.linalg.norm(self.error_history, axis=1))
        plt.axhline(0, color='r')
        plt.axhline(tol, color='g', linestyle='--')
        plt.title("Error History")
        plt.xlabel("Iteration")
        plt.ylabel("Pose Error (norm)")
        plt.show()

    def is_in_ellipsoid(self):
        """Check if the robot is within the manipulability ellipsoid"""
        self.FK()
        p = self.T[:3, 3].flatten()
        # Compute normalized-ellipsoid coordinate
        r2 = ((p[0]**2 + p[1]**2) / self.a_margin**2) + (p[2]**2 / self.c_margin**2)

        if r2 > 1.0:
            print(f'Robot is outside the manipulability ellipsoid.')
            self.stop = True
            return False
        
        return True
    

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
                    print(f"\nIK converged in {i} iterations.")
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
                    J_update = np.linalg.pinv(self.J).real
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
            print(f"\nIK did not converge within {max_iters} iterations. Final error norm: {np.linalg.norm(xi_e_space):.6f}")
            print("**********************************\n")
            return None

        finally:
            ## --- Restore the original state ---
            self.data.qpos[self.joint_idx] = q_original
            mujoco.mj_fwdPosition(self.model, self.data)
            print("IK finished, robot state restored.")
            print("**********************************")


    def diff_IK(self, v_des, damping):
        """
        Solve differential kinematics to achieve desired end-effector velocity
        v_des: 6D vector [lin_vel, ang_vel] in world frame
        Returns: joint velocities
        """
        # --- Update Mujoco state ---
        mujoco.mj_forward(self.model, self.data)
        # --- Compute Jacobian ---
        self.get_jacobian()

        # --- Compute velocity (twist) error---
        v_error = v_des - self.J @ self.data.qvel[self.joint_idx].reshape(-1,)

        # Damped Least Squares
        JT = np.linalg.pinv(self.J)
        dv = JT @ np.linalg.solve(self.J @ JT + (damping * np.eye(6)), v_error).reshape(6, 1)

        # Limit joint velocities
        vel_limit = 1.5 # rad/s
        dv = np.clip(dv, -vel_limit, vel_limit)

        return dv.reshape(-1)
    

    def get_pushing_force(self):
        """ Get the contact force on the payload from the pusher."""
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]

            # Only care about payload and pusher (skip table contact)
            if not (c.geom[0] == self.payload_geom_id or c.geom[1] == self.payload_geom_id ):
                continue
            other = c.geom[1] if c.geom[0] == self.payload_geom_id else c.geom[0]
            if other != self.pusher_geom_id:
                continue


             # 1) world‐frame contact force (first 3 entries of mj_contactForce)
            f6 = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, ci, f6)
            f_space = f6[:3]

            # 2) get EE frame rotation (3x3) and position(3,)
            R,_ = TransToRp(self.FK().copy())

            # 3) change frame Space -> EE
            f_ee = R.T @ f_space
            
            return f_ee.reshape(1, 3)
        
        return np.zeros((1, 3))  # No contact forces found
    

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
            [6*duration, 12*duration**2, 20*duration**3]
        ])
        
        b = np.array([
            q_end - q_start,
            np.zeros_like(q_start),
            np.zeros_like(q_start)
        ])
        
        # Solve A*x = b for x = [c3, c4, c5] for each joint
        c345 = np.linalg.solve(A, b)
        
        print(f"\nGenerated a {duration:.2f} sec trajectory to reach final pose.")
        return np.vstack([c0, c1, c2, c345])

    def evaluate_trajectory(self, t, coeffs, duration):
        """Evaluates the trajectory at a given time t."""
        if t < 0: t = 0
        if t > duration: t = duration
        return coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5
    

    def set_position_control(self, q_desired, damping=1e-4):
        """Apply position control to the robot"""
        # Check manipulability
        if not self.is_in_ellipsoid():
            print("Warning: Desired position is outside the manipulability ellipsoid.")
            return
        self.data.ctrl[self.joint_idx] = q_desired.reshape(6,1)
        mujoco.mj_forward(self.model, self.data) # Update forward kinematics after control input