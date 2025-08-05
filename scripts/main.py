# %%
# %load_ext autoreload
# %autoreload 2

# Set up GPU rendering.
import distutils.util
import os
import subprocess

# Configure MuJoCo to use the EGL rendering backend (requires GPU)
print('Setting environment variable to use GPU rendering:')
# %env MUJOCO_GL=egl

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
from scipy.spatial.transform import Rotation as Robj
from scipy.optimize import curve_fit, fsolve
from helper_fns import *
# import importlib
import robot_controller
# importlib.reload(robot_controller)

# Graphics and plotting.
import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)
# Set matplotlib font size
fonts = {'size' : 20}
plt.rc('font', **fonts)


## __________________________________________________________________________________

print("\nCWD: ", os.getcwd(), '\n')
model_path = 'assets/table_push.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

def set_render_opts(model, renderer):
        # tweak scales of contact visualization elements
        model.vis.scale.contactwidth = 0.025
        model.vis.scale.contactheight = 0.25
        model.vis.scale.forcewidth = 0.05
        model.vis.map.force = 0.3
        # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True # joint viz
        renderer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        renderer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        renderer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE # Show site frame(s)
        # Make site frame arrows smaller
        model.vis.scale.framewidth = 0.025
        model.vis.scale.framelength = .75
        # Set default camera position
        renderer.cam.distance = 2.5       # Distance from the camera to the scene
        renderer.cam.elevation = -30.0    # y-axis rotation
        renderer.cam.azimuth = 100.0      # z-axis rotation
        renderer.cam.lookat[:] = np.array([0.8, 0.0, 0.0])  # Center of the scene


# The set_render_opts function as it was defined, applied to mujoco.viewer
# For mujoco.Renderer, we will configure MjvCamera and MjvOption directly
# and pass them to update_scene.
def set_render_opts_for_renderer(model_obj, cam_obj, opt_obj):
    # Tweak scales of contact visualization elements (these apply to model.vis)
    model_obj.vis.scale.contactwidth = 0.025
    model_obj.vis.scale.contactheight = 0.25
    model_obj.vis.scale.forcewidth = 0.05
    model_obj.vis.map.force = 0.3
    model_obj.vis.scale.framewidth = 0.025
    model_obj.vis.scale.framelength = .75

    # Configure MjvOption flags
    opt_obj.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    opt_obj.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # To show frames, you typically enable a flag like mjVIS_JOINT, mjVIS_BODY, mjVIS_GEOM, mjVIS_SITE, etc.
    # and then set the specific frame type using model.vis.frame.
    # Removing mjVIS_FRAME as it does not exist.
    # opt_obj.flags[mujoco.mjtVisFlag.mjVIS_SITE] = True # Example: enable site visualization, which includes their frames
    opt_obj.frame = mujoco.mjtFrame.mjFRAME_SITE # Show site frame(s)


    # Configure MjvCamera
    cam_obj.distance = 2.5       # Distance from the camera to the scene
    cam_obj.elevation = -30.0    # y-axis rotation
    cam_obj.azimuth = 100.0      # z-axis rotation
    cam_obj.lookat[:] = np.array([0.8, 0.0, 0.0])  # Center of the scene
    


## Helper function to align y-axis limits of multiple axes to zero
def align_zeros(axes):
    ylims_current = {}   #  Current ylims
    ylims_mod     = {}   #  Modified ylims
    deltas        = {}   #  ymax - ymin for ylims_current
    ratios        = {}   #  ratio of the zero point within deltas

    for ax in axes:
        ylims_current[ax] = list(ax.get_ylim())
                        # Need to convert a tuple to a list to manipulate elements.
        deltas[ax]        = ylims_current[ax][1] - ylims_current[ax][0]
        ratios[ax]        = -ylims_current[ax][0]/deltas[ax]
    
    for ax in axes:      # Loop through all axes to ensure each ax fits in others.
        ylims_mod[ax]     = [np.nan,np.nan]   # Construct a blank list
        ylims_mod[ax][1]  = max(deltas[ax] * (1-np.array(list(ratios.values()))))
                        # Choose the max value among (delta for ax)*(1-ratios),
                        # and apply it to ymax for ax
        ylims_mod[ax][0]  = min(-deltas[ax] * np.array(list(ratios.values())))
                        # Do the same for ymin
        ax.set_ylim(tuple(ylims_mod[ax]))


## Nonlinear cosine fitting function
def nl_cos_fit(x, y, extend_factor=2, maxfev=10_000):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.shape != y.shape:
        print(f"x.shape: {x.shape}, y.shape: {y.shape}")
        raise ValueError("x and y must have the same shape.")
    N = len(x)
    dx = x[1] - x[0]  # Assume uniform spacing

    # 1) Define basic cos model with parameters to fit
    def cos_model(x, amp, omega, phi, offset):
        return amp * np.cos(omega * x + phi) + offset

    # 2) Initial guesses (nonlinear solvers need decent starting point)
    offset0 = np.mean(y)                    # init offset (vertical shift) -> data mean
    amp0 = (np.max(y) - np.min(y)) / 2      # init amplitude (peak2peak height) -> half the range
    phi0 = 0.0                              # init phase shift -> start cos at 0.0 (normal cos fn)

    # estimate omega from the FFT peak
    y0 = y - offset0                        # zero-centered (subtract offset so the zero-frequency component doesn't dominate)
    fft = np.fft.rfft(y0)                   # positive-frequency spectrum
    freq = np.fft.rfftfreq(len(x), d=1.0)   # tells each index's corresponding frequency in cycles per sample
    # ignore the zero-freq bin when finding the peak
    peak = np.argmax(np.abs(fft[1:])) + 1   # +1 because we ignored the zero-frequency bin
    omega0 = 2*np.pi*freq[peak]             # convert cycles/sample -> rad/sample

    # Initial guess parameter array
    p0 = [amp0, omega0, phi0, offset0]

    lower = [0.0, 0.0, -np.pi, -1.0] #-np.inf]
    upper = [np.inf, np.pi, np.pi, 1.0] #np.inf]

    ## Nonlinear least squares (!!__ THIS IS ACTUALLY LEVENBERG-MARQUARDT __!!) minimizes sum of sq errors
    # 3) Fit
    popt, _ = curve_fit(
        cos_model, x, y,
        p0=p0, bounds=(lower, upper),
        maxfev=maxfev)                          # popt is the optimal parameters. pcov is the covariance (diagonal is uncertainty of each parameter)
    
    # 4) Build extended x_out
    N_out = extend_factor * N
    x_start = x[0]
    x_out = x_start + np.arange(N_out) * dx

    y_out = cos_model(x_out, *popt)

    # Build fitted curve
    return x_out, y_out


## Let's also do a polynomial fit for comparison
def poly_fit(x, y, maxdeg=3, extend_factor=2):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.shape != y.shape:
        print(f"x.shape: {x.shape}, y.shape: {y.shape}")
        raise ValueError("x and y must have the same shape.")
    N = len(x)
    dx = x[1] - x[0]  # Assume uniform spacing

    # 1) Fit polynomial of degree maxdeg
    coeffs = np.polyfit(x, y, maxdeg)

    # 2) Build extended x_out
    N_out = extend_factor * N
    x_start = x[0]
    x_out = x_start + np.arange(N_out) * dx

    y_out = np.polyval(coeffs, x_out)

    return x_out, y_out


## Linear fit
def linear_fit(x, y, extend_factor=2):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.shape != y.shape:
        print(f"x.shape: {x.shape}, y.shape: {y.shape}")
        raise ValueError("x and y must have the same shape.")
    N = len(x)
    dx = x[1] - x[0]  # Assume uniform spacing

    # 1) Fit linear model
    coeffs = np.polyfit(x, y, 1)

    # 2) Build extended x_out
    N_out = extend_factor * N
    x_start = x[0]
    x_out = x_start + np.arange(N_out) * dx

    y_out = np.polyval(coeffs, x_out)

    return x_out, y_out


# Helper fn to shift pitch data to remove initial 'ramp up' of force
def shift_pitch_data(pitch_full, f_full):
    """
    Shift the pitch data to remove the initial 'ramp up' of force.
    This is done by finding the index of the maximum force and shifting the pitch data accordingly.
    """
    inflection_idx = np.argmax(f_full)
    n_copy = len(f_full) - inflection_idx
    temp_pitch = np.zeros_like(pitch_full)
    if n_copy > 0:
        temp_pitch[inflection_idx : inflection_idx + n_copy] = pitch_full[0 : n_copy]

    return temp_pitch.copy()  # Copy the shifted pitch data

# Fit a model to the data
def fit_model(f_full, pitch_full, t_full, theta_safe=np.inf, extend_factor=1.0, plot=True):
    
    f_full = np.linalg.norm(f_full, axis=1)     # Get the norm of the force vector

    # Mujoco oddly shows initially increasing force as two come in contact.
    # Let's shift the pitch data to remove the initial 'ramp up' of force, at f_max
    pitch_full = shift_pitch_data(pitch_full, f_full)

    t_rem = np.array([])
    f_rem = np.array([])
    pitch_rem = np.array([])
    
    t_sample = t_full
    f_sample = f_full
    pitch_sample = pitch_full

    zero_crossing_time = t_full[-1]
    topple_angle_est = pitch_full[-1]

    # ============================================================
    # If max_tip_angle specified, sample the data up to that angle
    # ============================================================
    if theta_safe != np.inf:
        # Find indexes where pitch is below theta_safe
        sample_idxs = np.where(np.abs(pitch_full) <= theta_safe)[0]
        # Also remove outliers before inflection_idx
        # sample_idxs = sample_idxs[sample_idxs > inflection_idx]  # Only keep indices

        # Now get the force, time, and pitch data up to that index
        f_sample = f_full[sample_idxs]
        t_sample = t_full[sample_idxs]
        pitch_sample = pitch_full[sample_idxs]


        ## FIRST TRY USING BASIC METHOD TO JUST GET F=0 IDX AND CORRESPONDING ANGLE

        force_coeffs = np.polyfit(t_sample, f_sample, deg=2)
        fitted_force_fn = np.poly1d(force_coeffs) # Create poly fn to evaluate force at any time

        # Define (python) function for fsolve
        def func_force(t):
            return fitted_force_fn(t)

        # Use fsolve to find the time when force is zero
        zero_crossing_time, = fsolve(func_force, t_sample[-1]*1.25) # Use initial guess just after last sampled t

        print(f"Zero crossing time: {zero_crossing_time:.3f} s")

        # Also extrapolate the pitch at that time
        pitch_coeffs = np.polyfit(t_sample, pitch_sample, deg=2)
        fitted_pitch_fn = np.poly1d(pitch_coeffs) # Create poly fn to evaluate pitch at any time
        topple_angle_est = fitted_pitch_fn(zero_crossing_time)
        print(f"Estimated topple angle at zero force crossing: {np.rad2deg(topple_angle_est):.3f} degrees")

        # Calculate the remaining time, force, and pitch data
        t_rem = np.linspace(t_sample[-1], zero_crossing_time, 100)
        f_rem = fitted_force_fn(t_rem)  # Fit the remaining force data using the fitted force function
        pitch_rem = fitted_pitch_fn(t_rem)  # Fit the remaining pitch data using the fitted pitch function

    # ===========================================================

    # 2) Let's fit using three different methods: nonlinear cosine fit , polynomial fit , and linear fit
    if plot:
        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax2 = plt.twinx()

        ax1.axhline(0, color='c', label='_') # Horizontal line at zero for reference

        # 3) Plot the sampled force data
        ax1.plot(t_sample, f_sample, color='b', linewidth=2, label='Push force (norm)')

        if len(t_rem) > 0:
            # Plot the extrapolated part
            ax1.plot(t_rem, f_rem, color='b', linestyle='--', label='_')

        # Plot the zero crossing point if applicable
        ax1.plot(zero_crossing_time, 0, 'kx', markersize=10, label='_')
        ax2.plot(zero_crossing_time, np.rad2deg(topple_angle_est), 'gx', markersize=10, label='_')

        ax1.set_xlabel("Time (s)", fontsize=20)
        ax1.set_ylabel("Force Norm (N)", color='b', fontsize=20)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_ylim(-0.1, 1.1 * np.max(f_full))
        ax1.set_xlim(0.9 * t_full[0], 1.1 * t_rem[-1] if len(t_rem) > 0 else 1.1 * t_full[-1])  # Set x-limits to the full range of times

        # 4) Plot the sampled payload pitch on ax2
        ax2.plot(t_sample, np.rad2deg(pitch_sample), color='g', linestyle='-', label='Payload pitch')      # Plot the payload tilt (pitch) angle
        ax2.plot(t_rem, np.rad2deg(pitch_rem), color='g', linestyle='--', label='_')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_ylabel("Payload Pitch (degrees)", color='g', fontsize=20)       # Add a second y-axis for the tilt angle
        ax2.set_ylim(-5, 30) # max hardcoded for now, can change to 1.1*max_pitch later
        
        align_zeros([ax1, ax2])  # Align the y-axes to zero

        # 4) Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=20)

        plt.title(r"Force & Pitch vs Time for $\theta_{safe}$ = %.f$^\circ$" %np.rad2deg(theta_safe), fontsize=20)
        plt.show()

    # print(zero_crossing_time, topple_angle_est)
    return zero_crossing_time, topple_angle_est


# Get the line-of-action passing thru CoM of generalized plane
def get_com(max_tip_angle, f, times, pitch, irb_controller, plot=True):
    """
    Estimate the center of mass (CoM) of the payload using the force, time, and pitch data.
    Instead of re-doing simulation, sample the data up to a maximum tip angle.
        max_tip_angle: Maximum tip angle in degrees to consider for the CoM estimation.
    """
    idx_sample = np.where(np.abs(pitch) <= np.deg2rad(max_tip_angle))[0]
    # Now get the force, time, and pitch data up to that index
    f_sample = np.linalg.norm(f, axis=1)[idx_sample]
    t_sample = times[idx_sample]
    pitch_sample = pitch[idx_sample]


    # ## Now let's estimate the CoM using sampled data.
    # # Recall the moment-balance equation for the payload:
    # # ΣM = 0 --> F * h_f = mg * (x_c * cos(pitch) - y_c * sin(pitch))
    # # Given that origin is taken at tipping/pivot edge

    # # Rearrange and convert to matrix form given N samples:
    # # F = A @ β, where A is [cos , -sin] matrix of shape (N, 2) and 
    # # β is [x_c; y_c] vector of shape (2,) scaled by the constant (mg/h_f)

    # # Create A = [-cos, -sin] matrix of pitch samples of shape (N, 2)
    # # Note: normally this would be [cos, -sin], but we take the origin at the pivot edge with x AWAY from rectangle
    # A = np.array([-np.cos(pitch_sample), -np.sin(pitch_sample)]).T

    # # Least-squares to find x vector (can also do np.linalg.lstsq)
    # beta = np.linalg.inv(A.T @ A) @ A.T @ f_sample        # Recall (A.T @ A)^-1 @A is the moore-Penrose pseudo-inverse
    # print(f"\nCoM lies on the line defined by: x_hat = {beta[0]:.3f}, y_hat = {beta[1]:.3f}")

    # # Calculate the height of the applied force on the payload WRT payload's frame (tipping edge)
    # # Height is (global) EE z-position minus the table surface (aka payload bottom) z-position
    # _, p_ee = TransToRp(irb_controller.FK())
    # h_f_global = p_ee[2]  # Global EE z-position
    # surface_height = irb_controller.get_surface_pos()[2]        # Table surface z-position
    # print(f"Surface_height: {surface_height:.2f} m, EE height: {h_f_global:.2f} m")
    # # payload_bottom = payload_pos - surface_ht                      # Table surface is 0.1 m, so bottom is at payload center z-position minus 0.1
    # h_f = h_f_global - surface_height                       # Height of applied force on the payload
    # m = 0.1 # Mass of the payload
    # g = 9.81 # Gravity
    # print(f"with h_f={h_f:.2f}, m={m}, g=9.81.")

    # x_c = beta[0] * h_f / (m * g)
    # y_c = beta[1] * h_f / (m * g)
    # print(f"Estimated CoM: x_c = {x_c:.3f}, y_c = {y_c:.3f}\n\n")

    # topple_angle_est = np.arctan2(x_c, y_c)  # atan2(y, x) gives the angle in radians
    # print(f"Estimated Toppling angle (manual LS): {np.rad2deg(topple_angle_est):.3f} degrees")


    ## NOW QUICKLY TRY USING BASIC METHOD TO JUST GET F=0 IDX AND CORRESPONDING ANGLE
    force_coeffs = np.polyfit(t_sample, f_sample, 1) # Linear at first
    fitted_force_fn = np.poly1d(force_coeffs) # Create poly fn to evaluate force at any time
    print(f"Fitted force fn: {fitted_force_fn}")

    # Find zero crossing of f
    initial_guess = t_sample[-1] * 1.1 # Start looking for zero crossing at the end of the data

    # Define (python) function for fsolve
    def func_force(t):
        return fitted_force_fn(t)

    # Use fsolve to find the time when force is zero
    from scipy.optimize import fsolve
    zero_crossing_time, = fsolve(func_force, initial_guess)

    print(f"Zero crossing time: {zero_crossing_time:.3f} s")

    # Also extrapolate the pitch at that time
    pitch_coeffs = np.polyfit(t_sample, pitch_sample, 1) # Linear at first
    fitted_pitch_fn = np.poly1d(pitch_coeffs) # Create poly fn to evaluate pitch at any time
    topple_angle_est = fitted_pitch_fn(zero_crossing_time)
    print(f"Estimated topple angle at zero force crossing: {np.rad2deg(topple_angle_est):.3f} degrees")
    

    if plot:
        ## Plot the sampled data and the fitted curves
        fit_model(f_sample, pitch, times, plot=True, extend_factor=22/max_tip_angle, plot_cospoly=False) # For plotting, provide the entire pitch history

    return x_c, y_c, topple_angle_est




# %% [markdown]
# ## Full simulation of toppling

# %%
## Let's recall the model to reset the simulation
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
joints = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

## Setup based on robot model
irb_controller = robot_controller.controller(model, data, joint_names=joints, ee_site='ee_site', table_site='surface_site')

## We know the following pose is just in front of the payload
T_init = np.array([[    0.0,   0.0,     1.0,     0.93 ],
                    [   0.0,   1.0,     0.0,     0.0  ],
                    [  -1.0,   0.0,     0.0,     0.25 ],
                    [   0.0,   0.0,     0.0,     1.0  ]])

## Set the robot to the desired initial pose
# print("Method 2: Damped Least Squares")
sol_Damped_LS = irb_controller.IK(np.array(T_init), method=2, damping=0.5, max_iters=1000)
irb_controller.set_pose(sol_Damped_LS)

## FOR VELOCITY CONTROL (format: [vx vy vz wx wy wz])
target_vel = np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0])

## Initialize force and tilt history for plotting
f_hist = np.zeros((0,3))
pitch_hist = np.empty((0,))
# Setup time step history
t_hist = np.empty((0,))
data.time = 0.0

# Set the stop angle, for full sim this is when the payload topples over (i.e. pitch angle is 90 degrees)
STOP_ANGLE = np.deg2rad(90.0)

# Array of the tipping edge 'line segment'
# NOTE: code currently only allows for line-segment contacts aka two-point contacts, not single-point or multi-point.
tip_edge_pos = []

# --- Additions for video recording ---
frames = []
framerate = 30
duration = 10

# with mujoco.viewer.launch_passive(model, data, show_left_ui=False) as viewer:
# Create MjvCamera and MjvOption objects
sim_cam = mujoco.MjvCamera() # This will be our camera for rendering
sim_opt = mujoco.MjvOption() # This will be our visualization options

# Initialize them with default values (important!)
mujoco.mjv_defaultCamera(sim_cam)
mujoco.mjv_defaultOption(sim_opt)
set_render_opts_for_renderer(model, sim_cam, sim_opt)  # Set the rendering options

with mujoco.Renderer(model, height=720, width=1280) as renderer:
    # set_render_opts(model, renderer)
    
    # while viewer.is_running() and not irb_controller.stop:
    while not irb_controller.stop:
        # Set joint velocities
        irb_controller.set_velocity_control(target_vel)

        # Get payload pitch and check toppling stop condition
        tip_angle = irb_controller.get_payload_pose(output='pitch')

        if abs(tip_angle - STOP_ANGLE) < 1e-3:
            print("Payload has tipped to threshold, stopping simulation.")
            irb_controller.stop = True

        # Get contact force(s) ONLY between EE and payload
        f_curr = irb_controller.get_pushing_force()
        f_curr_norm = np.linalg.norm(f_curr)

        # Initialize contact vertices list
        contact_vertices = []

        # Only append if contact is occurring with pusher and payload AND force is decreasing
        if f_curr_norm > 0.0:
            # Append current force, pitch angle, and time to history
            f_hist = np.vstack((f_hist, f_curr))
            pitch_hist = np.append(pitch_hist, tip_angle)
            t_hist = np.append(t_hist, data.time)

            ## We would like to define a plane on which the CoM lies. A plane can be defined by two lines:
            # In this case, one line is the contact edge of payload and table, and the second line is the 
            # line-of-action of critical toppling angle
            # If so, get the two contact points of tipping edge
            for contact in irb_controller.data.contact:
                geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(id)) for id in contact.geom]
                # If the contact is between the pusher and payload, skip it
                if 'push_rod' in geom_names:
                    continue
                else:
                    contact_vertices.append(contact.pos)
        
        # IFF there are exactly two contacts between the payload and table, we can define a line segment in the plane
        if len(contact_vertices) == 2:
            tip_edge_pos.append(np.array(contact_vertices).ravel())  # Append the contact vertices as a row vector

        # Update the viewer
        mujoco.mj_step(model, data)
        
        # viewer.sync()

        if len(frames) < data.time * framerate:
            renderer.update_scene(data, camera=sim_cam, scene_option=sim_opt)  # Update the renderer with the current scene
            pixels = renderer.render()
            frames.append(pixels)  # Capture the current frame for video recording

# %%

media.show_video(frames, fps=60, loop=True)

# %% [markdown]
# ## Fit an approximation to the full force data

# %%
## Here's the actual CoM position in the payload frame
# We assume we have the x,y position from 2D com, but the z position is unknown.
c_proj = np.array([-0.04, 0.04, 0.1])
print(f"========\nActual CoM Position in payload frame: {c_proj}\n========")

t_ZMP, theta_crit = fit_model(f_hist, pitch_hist, t_hist, plot=True)
print(f"\nZero-crossing time: {t_ZMP:.3f} s")
print(f"\nTOPPLING (CRITICAL) ANGLE: {np.rad2deg(theta_crit):.2f} degrees\n")
# Actual toppling angle = 21.8 degrees (found thru: theta = atan(r_w/h_c))

## Get avg tip edge vertices
tip_edge_avg = np.mean(np.array(tip_edge_pos), axis=0)
p1 = tip_edge_avg[:3]  # First three elements are the first contact point
p2 = tip_edge_avg[3:]  # Last three elements are the second contact point

## ____ Three Ways to estimate CoM ____ ##

## 1) Using projection of 2D xy CoM onto the plane defined by the tip edge and the critical toppling angle
# The plane normal:
n = np.array([np.cos(theta_crit), 0, np.sin(theta_crit)]) # Found by 'rotating' the y-z plane by toppling angle

# The projection of CoM onto the above plane
cz = -(c_proj[0]*n[0] + c_proj[1]*n[1]) / n[2] # z-coordinate of the CoM projection onto the plane
c = np.array([c_proj[0], c_proj[1], cz])  # CoM projection onto the plane
print(f"\nEstimated CoM Method 1: {c}\n")


## 2) Using the critical toppling angle and the distance from CoM projection to tipping edge
# Define remaining vectors:
v_tip = p2 - p1  # Vector along the line segment (tipping edge)
w = c_proj - p1  # Vector from the first contact point to the CoM projection

# Euclidian (norm) distance from CoM_proj to tipping edge
d = np.linalg.norm(np.cross(v_tip, w)) / np.linalg.norm(v_tip)
x0 = c_proj[0]
y0 = c_proj[1]
x1 = p1[0]
x2 = p2[0]
y1 = p1[1]
y2 = p2[1]

# Calculate d more explicitly
d_alternate = np.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
# d should be around 0.04 m but is being calculated as 1.09


# quickly get the position of the box frame at site 'frame_site'
box_pos = irb_controller.get_payload_pose(site='frame_site')
# print(f"Box frame position: {box_pos}")
# Calculate the CoM height zc using critical toppling angle
zc = d / np.tan(theta_crit)
# And the final 3d CoM is:
com = np.array([c_proj[0], c_proj[1], zc])
print(f"\nEstimated CoM Method 2: {com}\n")


## 3) Use the scalar def and geometry and tan to find CoM
# first find r_c distance from tipping edge to CoM (radius of CoM rotation)
r_c = 0.04
h_c = r_c / np.tan(theta_crit)
# h_c = r_c / np.tan(np.deg2rad(21.8014))
print(f'\nEstimated CoM Method 3: {np.array([c_proj[0], c_proj[1], h_c])}\n')

# %% [markdown]
# ### Also plot pitch vs force

# %%
f_hist_norm = np.linalg.norm(f_hist, axis=1)
pitch_hist_deg = shift_pitch_data(np.rad2deg(pitch_hist), f_hist_norm)
plt.plot(pitch_hist_deg, f_hist_norm, label='Sampled data')
plt.xlabel('Payload Pitch (degrees)')
plt.ylabel('Force Norm (N)')
plt.title('Payload Pitch vs Force Norm')
plt.legend()
plt.show()

# %% [markdown]
# ### Now to do a small push and interpolate the force zero-crossing
# 
# Instead of re-simulating, take a subset of the full data

# %%
# Let's choose a payload angle at which to 'sample' our data and find that index

# theta_safe = 6 # degrees
_, theta_crit = fit_model(f_hist, pitch_hist, t_hist, theta_safe=np.deg2rad(6), plot=True)
z_c = 0.04 / np.tan(theta_crit)  # Calculate the height of CoM using the critical angle
print(f"Estimated CoM height at theta_safe=6 degrees: {z_c:.3f} m\n\n")

# theta_safe = 12 # degrees
_, theta_crit = fit_model(f_hist, pitch_hist, t_hist, theta_safe=np.deg2rad(12), plot=True)
z_c = 0.04 / np.tan(theta_crit)  # Calculate the height of CoM using the critical angle
print(f"Estimated CoM height at theta_safe=12 degrees: {z_c:.3f} m\n\n")

# theta_safe = 18 # degrees
_, theta_crit = fit_model(f_hist, pitch_hist, t_hist, theta_safe=np.deg2rad(18), plot=True)
z_c = 0.04 / np.tan(theta_crit)  # Calculate the height of CoM using the critical angle
print(f"Estimated CoM height at theta_safe=18 degrees: {z_c:.3f} m\n\n")

# %% [markdown]
# ### Some options and extensions:
# 
# Before pushing:
# - Determine 'good' pushing face
# 
# Additional Parameters to Extract:
# - Inertia matrix
# - Mass
# - Friction
# 
# During pushing:
# - Stop tipping at 50% of initial pushing contact force
# 
# After pushing:
# - Push along a trajectory
# 
# Separate extensions:
# - Compare against adaptive control parameter adjustment
# - Use LLM to determine good pushing face
# - Train RL and generate 3D NONPREHENSILE data set

# %% [markdown]
# 


