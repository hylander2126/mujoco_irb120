# %% [markdown]
# ![MuJoCo banner](https://raw.githubusercontent.com/google-deepmind/mujoco/main/banner.png)
# 
# # <h1><center>Tutorial  <a href="https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" width="140" align="center"/></a></center></h1>
# 
# This notebook provides an introductory tutorial for [**MuJoCo** physics](https://github.com/google-deepmind/mujoco#readme), using the native Python bindings.
# 
# <!-- Copyright 2021 DeepMind Technologies Limited
# 
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
# 
#          http://www.apache.org/licenses/LICENSE-2.0
# 
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
# -->

# %% [markdown]
# # All imports

# %%
# Set up GPU rendering.
import distutils.util
import os
import subprocess
# # if subprocess.run('nvidia-smi').returncode:
# #   raise RuntimeError(
# #       'Cannot communicate with GPU. '
# #       'Make sure you are using a GPU Colab runtime. '
# #       'Go to the Runtime menu and select Choose runtime type.')

# # Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
# # This is usually installed as part of an Nvidia driver package, but the Colab
# # kernel doesn't install its driver via APT, and as a result the ICD is missing.
# # (https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)
# NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
# if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
#   with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
#     f.write("""{
#     "file_format_version" : "1.0.0",
#     "ICD" : {
#         "library_path" : "libEGL_nvidia.so.0"
#     }
# }
# """)

# Configure MuJoCo to use the EGL rendering backend (requires GPU)
print('Setting environment variable to use GPU rendering:')
%env MUJOCO_GL=egl

# Check if installation was succesful.
try:
  print('Checking that the installation succeeded:')
  import mujoco
  mujoco.MjModel.from_xml_string('<mujoco/>')
  import mujoco.viewer # Also have to import this to trigger the installation of the viewer.
except Exception as e:
  raise e from RuntimeError(
      'Something went wrong during installation. Check the shell output above '
      'for more information.\n'
      'If using a hosted Colab runtime, make sure you enable GPU acceleration '
      'by going to the Runtime menu and selecting "Choose runtime type".')

print('Installation successful.')

# Other imports and helper functions
import time
import itertools
import numpy as np
from helper_fns import *

# Graphics and plotting.
import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# %% [markdown]
# # Function and Class Setup
# 

# %%
model_path = '../assets/environments/table_push.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

class controller:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, joint_names, site_name):
        self.model = model
        self.data = data
        # self.viewer = viewer
        self.q_indices = np.array([model.joint(name).qposadr for name in joint_names])
        self.v_indices = np.array([model.joint(name).dofadr for name in joint_names])
        self.ee_site = model.site(site_name).id
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.joint_names = joint_names
        self.n_joints = len(joint_names)
        self.J = np.zeros((6, self.n_joints))

        # Get joint limits
        self.q_min = model.jnt_range[self.q_indices, 0]
        self.q_max = model.jnt_range[self.q_indices, 1]

    def FK(self):
        """
        Forward kinematics via Mujoco
        Returns: 
            T: 4x4 homogeneous transform of end effector
        """
        # Update position
        mujoco.mj_fwdPosition(self.model, self.data)
        # Assemble FK from Mujoco state info
        R_curr = self.data.site_xmat[self.ee_site].reshape(3, 3)
        p_curr = self.data.site_xpos[self.ee_site].reshape(3, 1)
        T_curr = np.eye(4)
        T_curr[:3, :3] = R_curr
        T_curr[:3, 3] = p_curr.flatten()
        return T_curr
    
    def IK(self, T_des, method=2, max_iters=500, tol=1e-4, damping=1e-3):
        """
        Pose IK via numerical Damped-LS on twist error
            T_des: Desired end effector pose as a 4x4 homogeneous transform
        Returns: 
            q: joint angles 
        """
        # Check that T_des is a valid homogeneous transform
        if T_des.shape != (4, 4):
            raise ValueError("T_des must be a 4x4 homogeneous transform matrix.")
        
        # __ Initialize from Mujoco state __
        q = self.data.qpos[self.q_indices].copy()
        
        for i in range(max_iters):
            # Update position
            self.data.qpos[self.q_indices] = q
            mujoco.mj_fwdPosition(self.model, self.data)
            # Get current pose from Mujoco FK
            T_curr = self.FK()
            # Get Jacobian
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.ee_site)

            # Compute error transform in SE(3) AKA ("Where to move from T_curr to reach T_des?")
            T_err = np.linalg.inv(T_curr) @ T_des # "undo" current pose, then apply desired pose
            
            # Log of that relative motion to get the single true error-screw
            M_err = MatrixLog6(T_err) # 4x4 se(3)
            # Extract associated screw [ω(3); v(3)] from se(3) matrix
            xi_err_body = np.concatenate([np.array([M_err[2, 1], M_err[0, 2], M_err[1, 0]]),  M_err[:3, 3]]).reshape(6,1) # from skew matrix
            xi_err = twistbody2space(xi_err_body, T_curr)
            
            if np.linalg.norm(xi_err) < tol:
                print(f"\nIK converged in {i} iterations.")
                break
            elif i == max_iters - 1:
                print(f"\nIK failed to converge after {max_iters} iterations.")
                print(f"Final error: {xi_err}")
                break
             
            # Build the 6xn Jacobian
            J_pos = self.jacp[:, self.v_indices]
            J_rot = self.jacr[:, self.v_indices]
            self.J = np.vstack([J_rot, J_pos]).squeeze()

            J_inv = np.linalg.pinv(self.J)

            # Choose update method (1: Newton-Raphson, 2: Damped LS, 3: Gradient Descent))
            if method == 1:
                # Newton-Raphson
                delta_q = J_inv @ xi_err
            
            elif method == 2:
                # Damped least squares (Levenberg-Marquardt)
                term_to_invert = self.J @ self.J.T + (damping**2) * np.eye(6)
                delta_q = self.J.T @ np.linalg.solve(term_to_invert, xi_err) # equivalent to inv(A) @ xi_err
                # delta_q *= 0.3

            elif method == 3:
                # Gradient descent
                delta_q = J_inv @ xi_err
                # Scale by damping factor
                delta_q *= damping

            else:
                raise ValueError("Invalid method. Choose 1, 2, or 3.")

            # Make sure dq is a column vector
            delta_q = delta_q.reshape(6, 1)

            q += delta_q

        # Clamp joint angles to limits
        q = np.clip(q, self.q_min, self.q_max)
        return q

    def diff_IK(self, v_des, damping):
        """
        Solve differential kinematics to achieve desired end-effector velocity
        v_des: 6D vector [lin_vel, ang_vel] in world frame
        Returns: joint velocities
        """
        # Update position and get Jacobian
        mujoco.mj_fwdPosition(self.model, self.data)
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.ee_site)

        # Extract Jacobian for our joints
        J_pos = self.jacp[:, self.v_indices]
        J_rot = self.jacr[:, self.v_indices]
        self.J = np.vstack([J_pos, J_rot]).squeeze()

        # Compute velocity error
        v_error = v_des - self.J @ self.data.qvel[self.v_indices].reshape(-1,)

        # Damped Least Squares
        JT = np.linalg.pinv(self.J)
        dv = JT @ np.linalg.solve(self.J @ JT + (damping * np.eye(6)), v_error).reshape(6, 1)

        # Limit joint velocities
        vel_limit = 1.5 # rad/s
        dv = np.clip(dv, -vel_limit, vel_limit)
        
        return dv.reshape(-1)
    
    def set_velocity_control(self, v_desired, damping=1e-4):
        """Apply velocity control to the robot"""
        # if np.linalg.matrix_rank(self.J) < 6:
        #     # singular or near-singular configuration
        #     dq = np.zeros(self.n_joints)
        # else:
        dq = self.diff_IK(v_desired, damping)
        
        self.data.ctrl[:] = dq.reshape(-1)

    
    def reset_pose(self, home_pos=np.zeros((6,1))):
        """Reset robot to desired position"""
        # Set qpos to home position
        self.data.qpos[self.q_indices] = home_pos
        self.data.qvel[self.v_indices] = 0.0
        mujoco.mj_fwdPosition(self.model, self.data)
        return self.data.qpos[self.q_indices].copy()



def set_render_opts(model, viewer):
        # tweak scales of contact visualization elements
        model.vis.scale.contactwidth = 0.025
        model.vis.scale.contactheight = 0.25
        model.vis.scale.forcewidth = 0.05
        model.vis.map.force = 0.3
        # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True # joint viz
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        # Set default camera position
        viewer.cam.distance = 2.5       # Distance from the camera to the scene
        viewer.cam.elevation = -30.0    # y-axis rotation
        viewer.cam.azimuth = 100.0      # z-axis rotation
        viewer.cam.lookat[:] = np.array([0.8, 0.0, 0.0])  # Center of the scene


# %%
## Let's recall the model to reset the simulation
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
joints = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

# Setup based on robot model
irb_controller = controller(model, data, joint_names=joints, site_name='ee_site')


with mujoco.viewer.launch_passive(model, data) as viewer:
    set_render_opts(model, viewer)

    # Print the current EE pose transformation matrix
    print('Home Configuration:\n', irb_controller.FK())

    ## Through trial and error, we know the following joint configuration yields our desired initial pose:
    q_init = np.array([0.0, 0.525, 0.816, 0.0, -1.33, 0.0]).reshape(6, 1)
    print('Initial Joint Configuration:\n', q_init.flatten())
    ## Which corresponds to the following pose:
    irb_controller.reset_pose(q_init)
    T_init = irb_controller.FK()
    print("Initial Pose:\n", T_init)
#     Initial Pose:
#  [[-0.011  0.     1.     0.844]
#  [ 0.     1.     0.     0.   ]
#  [-1.     0.    -0.011  0.245]
#  [ 0.     0.     0.     1.   ]]
    
    ## Let's reset our pose to the home configuration
    irb_controller.reset_pose()

    ## Now we want to see if our IK solution can get this same joint configuration given the desired pose
    q_target = irb_controller.IK(np.array(T_init), method=2, damping=1e-3)
    ## Check that the IK joint configuration is the same as the initial joint configuration
    print("IK Joint Configuration:\n", q_target.flatten())
    ## Which corresponds to the following pose:
    irb_controller.reset_pose(q_target)
    T_target = irb_controller.FK()
    print("Target Pose:\n", T_target)
# IK Joint Configuration:
#  [ 0.     1.92   1.92   0.    -2.094  0.   ]
# Target Pose:
#  [[-0.985  0.    -0.174  0.465]
#  [ 0.     1.     0.     0.   ]
#  [ 0.174  0.    -0.985  0.267]
#  [ 0.     0.     0.     1.   ]]

## CLEARLY, these are not the same poses.

    ## FOR VELOCITY CONTROL (format: [vx vy vz wx wy wz])
    # target_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    while viewer.is_running():        
        # irb_controller.set_velocity_control(target_vel)

        mujoco.mj_step(model, data)
        viewer.sync()



