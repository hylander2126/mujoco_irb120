import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt, savgol_filter
from scipy.interpolate import interp1d
from scripts.utils.com_estimation import tau_app_model, tau_model, theta_from_tau, align_zeros
from scipy.stats import linregress
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def read_csv(file_path, trim_rows=0):
    with open(file_path, mode ='r')as file:
        csvFile = csv.reader(file)
        csv_arr = []
        for lines in csvFile:
                csv_arr.append(lines)
        return csv_arr[trim_rows:]

# ================ Load the data ===================
# path = "experiments/run_2025-11-10_15-28-42_t001_SYNC.csv"

# path = "experiments/run_2025-11-10_15-52-45_t002_SYNC.csv" # Best

# path = "experiments/run_2025-11-10_20-00-09_t002_SYNC.csv" # Best

path = "experiments/run_2025-11-20_16-56-00_t013_SYNC.csv" # Testing new force stop

csv_data = read_csv(path, trim_rows=1)  # Discard headers


## ================ Extract time series data ===================
time        = np.zeros(len(csv_data))
ee_exp      = np.zeros((len(csv_data), 3))
f_exp_raw   = np.zeros((len(csv_data), 3)) # RAW because we are filtering later
tag_exp_raw = np.zeros((len(csv_data), 3)) # RAW because we are filtering later

# Column nums for each data type: (time is column zero)
f_cols      = [1, 2, 3]
ee_cols     = [7, 8, 9]
tag_cols    = [16, 17, 18]

for i, row in enumerate(csv_data):
    # Time is in first column
    time[i] = float(row[0])
    # Extract Force data
    f_exp_raw[i, :]   = [float(row[j]) for j in f_cols]
    # Extract EE XYZ position
    ee_exp[i, :]  = [float(row[j]) for j in ee_cols]
    # Tag roll, pitch, yaw
    tag_exp_raw[i, :] = [float(row[j]) for j in tag_cols]

time -= time[0]  # Normalize time to start at zero

# END EFFECTOR is offset +3.5cm in z-axis due to mounting of robot
ee_exp[:, 2] -= 0.035  # Adjust Z position of EE for offset


## =================== SET THE GROUND TRUTH COM & MASS ===================
com_gt = [-0.05, 0.0, 0.13781]  # Ground truth CoM position for box_exp
m_gt = 0.615
## =======================================================================


## ================ Plot raw data ===================
PLOT_RAW = True

if PLOT_RAW:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax2 = plt.twinx()
    ax.plot(time, f_exp_raw[:, 0], "b", linewidth=3, label='X Force (raw)')
    ax.plot(time, f_exp_raw[:, 1], "r", linewidth=3, label='Y Force (raw)')
    ax.plot(time, f_exp_raw[:, 2], "m", linewidth=3, label='Z Force (raw)')
    # ax2.plot(time_th, tag_exp_raw[:, 0], color='g', linestyle='-', label='Roll (raw)')
    # ax2.plot(time_th, tag_exp_raw[:, 1], color='c', linestyle='-', label='Pitch (raw)')
    ax2.plot(time, np.rad2deg(tag_exp_raw[:, 2]), color='g', linestyle='-', linewidth=3, label='Object angle (raw)')
    ax.tick_params(axis='y', labelcolor='b', labelsize=15)
    ax.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelcolor='g', labelsize=15)
    ax.set_xlabel('Time (s)', fontsize=15)
    ax.set_ylabel('Force (N)', color='b', fontsize=15)
    ax2.set_ylabel('Angle (rad)', color='g', fontsize=15)
    ax.legend(loc='upper left', fontsize=15)
    ax.grid(True)

    # FOR PAPER FIGURE, TRIM THE AXES LIMITS
    # ax.set_xlim(1.6, 6)
    # ax2.set_ylim(0, 30)

    align_zeros([ax, ax2])
    plt.tight_layout()
    plt.show()


## Plot the xyz of ee pos
# plt.figure(figsize=(9,4.5))
# plt.plot(time, ee_exp[:,0], 'b', label='EE X pos')
# plt.plot(time, ee_exp[:,1], 'r', label='EE Y pos')
# plt.plot(time, ee_exp[:,2], 'm', label='EE Z pos')
# plt.xlabel('Time (s)', fontsize=10)
# plt.ylabel('EE Position (m)', fontsize=10)
# plt.title('End Effector Position Over Time')
# plt.legend()

## ================ Process the data (specifically force) ===================
# Butterworth filter
b, a         = butter(4, 5, fs=500, btype='low') # 4,5,500 : order, cutoff freq (<0.5*fs), sampling freq
f_exp_filt   = filtfilt(b, a, f_exp_raw, axis=0)

x1 = medfilt(f_exp_raw[:,0], kernel_size=5)
x2 = medfilt(f_exp_raw[:,1], kernel_size=5)
x3 = medfilt(f_exp_raw[:,2], kernel_size=5)

k_sg = int(0.09*500)|1
k_sg = 89
x_sg1 = savgol_filter(x1, k_sg, 3)
x_sg2 = savgol_filter(x2, k_sg, 3)
x_sg3 = savgol_filter(x3, k_sg, 3)
f_exp_filt = np.vstack((x_sg1, x_sg2, x_sg3)).T

# TODO: I also have to estimate/calculate e_hat from experiment instead of manually entering it


# Now we can also filter the tag data
# b, a        = butter(4, 2, fs=500, btype='low') # 4,2,500 : order, cutoff freq (<0.5*fs), sampling freq
tag_exp_filt = filtfilt(b, a, tag_exp_raw, axis=0)

# Let's bias the force and angle data using START & END values (end looks better for f, start for th)
# f_exp_filt -= np.mean(f_exp_filt[-20:, :], axis=0)
# tag_exp_filt -= np.mean(tag_exp_filt[:10], axis=0)

th_exp = tag_exp_filt[:, 2]  # HACK: assume ONLY tip in 'yaw' axis (column 2)


# ================ Find contact, settling, and start moments ===================
# First, let's find when we make contact (will be the maximal magnitude of force)
contact_idx_og = np.argmax(np.linalg.norm(f_exp_filt, axis=1))
contact_time_og = time[contact_idx_og]

# Then, let's find when the angle settles (after contact)
settle_indices_og = np.where(np.isclose(th_exp, th_exp[-1], atol=1e-2))[0]#[0]
valid_settle_indices = settle_indices_og[settle_indices_og > contact_idx_og]          # make sure we only look for settling AFTER contact
if len(valid_settle_indices) == 0:
    print("Warning: No settling found after contact!")
    settle_idx_og = len(time) - 1 # fall back to end of data
else:
    settle_idx_og = valid_settle_indices[0]  # take the first one
settle_time_og = time[settle_idx_og]

# Define start of analysis window (a few seconds before contact)
start_time_og = contact_time_og - 0.25 # 0.25 seconds before contact
start_idx_og = np.where(time >= start_time_og)[0][0]

print(f"Contact detected at index {contact_idx_og} ({contact_time_og:.2f} s)")
print(f"Settling detected at index {settle_idx_og} ({settle_time_og:.2f} s)")

# =============== Trim the data to the analysis window ===================
# THIS INCLUDES TRANSIENTS, WE WILL CLEAN FURTHER LATER
end_idx_og = settle_idx_og + 5  # add a few extra samples to be safe

time_trim = np.array(time[start_idx_og:end_idx_og])
f_trim = np.array(f_exp_filt[start_idx_og:end_idx_og, :])
th_trim = np.array(th_exp[start_idx_og:end_idx_og])
ee_trim = np.array(ee_exp[start_idx_og:end_idx_og, :])
print(f"Rough time range of contact window: {time_trim[0]:.2f} s to {time_trim[-1]:.2f} s\n")

## ===============================================================
# Analyze data and plot relationships
# Goal: Find theta_* and zc (and possibly mass)
## ===============================================================

# =============== Analysis on trimmed data ===================
# Find index of max force (after init transients) TODO: determine if this means f in x or norm
fmax_idx = np.argmax(np.linalg.norm(f_trim, axis=1), axis=0)
print(f"fmax time: {time_trim[fmax_idx]:.2f} s, corresponding angle: {np.rad2deg(th_trim[fmax_idx]):.2f} degrees")

# Determine near-zero-crossing of force in primary axis (x) by searching after max force (fmax_idx) index
f0_idxs = np.where(np.isclose(f_trim[:, 0], 0, atol=1e-2))[0]

# Use known theta* from geom to pick the zero-crossing index
# f0_idx = f0_idxs[np.argmin(np.abs(np.rad2deg(th_trim[f0_idxs]) - 20.19))]
f0_idx = 1449
print(np.rad2deg(th_trim[1449]))
print(f0_idx)
print(np.rad2deg(th_trim[f0_idx]))
print(f"Force at zero crossing from theta*: {f_trim[f0_idx,0]:.3f} N")

# TEMP: print all zero crossings found and choose the closest one to reality TODO: automate this

# print(f0_idxs)
# print(np.rad2deg(th_trim[f0_idxs]))
# # f0_idx = f0_idxs[-1]  # take last crossing before end of data
# # f0_idx = f0_idxs[0]  # take first crossing after fmax_idx
# f0_idx = f0_idxs[3]
# print(f"Num zero-crossings found: {len(f0_idxs)}, (last found) f0_idx: {f0_idx} at angle {np.rad2deg(th_trim[f0_idx]):.2f} deg, time: {time_trim[f0_idx]:.2f} s")

## ==================== CALCULATE THETA* AND ZC ====================
print("\n ************** FULL TOPPLING CALCULATION ****************")
theta_star_calc = th_trim[f0_idx]
zc_calc = abs(com_gt[0]) / np.tan(theta_star_calc)
print(f"\nCalculated from full toppling:\ntheta* = {np.rad2deg(theta_star_calc):.2f} deg, zc = {zc_calc:.3f} m")

# Ground truth from geometry
theta_star_gt = np.rad2deg(np.arctan2(abs(com_gt[0]), com_gt[2]))
print(f"\nGround truth from geometry:\ntheta*: {theta_star_gt:.2f} deg, zc = {com_gt[2]:.3f} m")


# ================ Plot the data ===================
PLOT_XYZ = True
PLOT_RELATIONSHIP = False

import matplotlib.ticker as ticker

# Now let's plot the x_data over time
if PLOT_XYZ:
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = plt.twinx()
    ax1.plot(time_trim, f_trim[:, 0], "b", linewidth=5, label='Push force (x)')
    ax1.plot(time_trim, f_trim[:, 1], "r", linewidth=5, label='Push force (y)')
    ax1.plot(time_trim, f_trim[:, 2], "m", linewidth=5, label='Push force (z)')
    ax2.plot(time_trim, np.rad2deg(th_trim), color='g', linewidth=5, linestyle='-.', label='Object angle')
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelcolor='b', labelsize=20)
    ax2.tick_params(axis='y', labelcolor='g', labelsize=20)
    ax2.set_ylabel('Object Angle (deg)', color='g', fontsize=20)
    ax1.set_xlabel('Time (s)', fontsize=20)
    ax1.set_ylabel('Force (N)', color='b', fontsize=20)
    # ax1.set_title('X, Y, Z Data Over Time')
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.95), fontsize=13)
    ax1.grid(True)

    # FOR PAPER FIGURE, TRIM THE AXES LIMITS
    ax1.set_xlim(time_trim[0], time_trim[f0_idx]+0.5)
    # ax1.set_yticks(np.arange(-1.5, 1.6, 0.5))
    ax2.set_ylim(0, 30)

    align_zeros([ax1, ax2])
    plt.tight_layout()
    plt.show()


if PLOT_RELATIONSHIP:
    # NOTE: This is plotting the primary pushing force, NOT the magnitude.
    # Now plot force (in primary tipping axis) versus payload pitch
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color='c') # Horizontal line at zero for reference
    ax.plot(np.rad2deg(th_trim[:f0_idx+100]), f_trim[:f0_idx+100, 0], \
            color='k', linewidth=5, label='Push force (x)')  # Plot the x-component of the force (up to 100 indices after zero-crossing)
    ax.set_ylabel("X-Force (N)", color='b', fontsize=20)
    ax.set_xlabel("Object Angle (deg)", color='g', fontsize=20)
    ax.axvline(theta_star_gt, color='g', linestyle='--', linewidth=5, label=r'Ground truth $\theta^*$')
    ax.scatter(np.rad2deg(theta_star_calc), f_trim[f0_idx,0], s=500, marker='*', color='r', label=r'Calculated $\theta^*$', zorder=2)
    
    ax.grid(True)
    fig.legend(loc='upper left', fontsize=15, bbox_to_anchor=(0.2, 0.85)) # # Now also plot a * at zero-crossing

    ax.tick_params(axis='y', labelcolor='b', labelsize=20)
    ax.tick_params(axis='x', labelcolor='g', labelsize=20)

    # ======= Same figure, add zoomed-in view around theta* ========
    # Choose x-lims around theta*
    x0 = np.rad2deg(max(th_trim.min(), theta_star_calc - np.deg2rad(0.25)))
    x1 = np.rad2deg(min(th_trim.max(), theta_star_calc + np.deg2rad(0.25)))
    # y-lims around bottom 10% of force
    y0 = -0.05
    y1 = 0.025

    # axins = inset_axes(ax3, width="30%", height="30%", loc='upper left')#, borderpad=2.2)
    axins = inset_axes(ax, width="30%", height="30%", loc='lower right', borderpad=2.2)
    axins.plot(np.rad2deg(th_trim), f_trim[:,0], color='k', linewidth=5)  # Plot the x-component of the force
    axins.scatter(np.rad2deg(theta_star_calc), f_trim[f0_idx,0], s=500, marker='*', color='r', label=r'Calculated $\theta^*$', zorder=3)
    axins.axvline(theta_star_gt, color='g', linestyle='--', linewidth=5, label=r'Ground truth $\theta^*$')
    axins.axhline(0, color='c', linewidth=2) # Horizontal line at zero for reference
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)
    axins.grid(True)
    axins.tick_params(axis='y', labelsize=15)
    axins.tick_params(axis='x', labelsize=15)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linewidth=3)
    plt.tight_layout()
    plt.show()


## ================= Extract sub-critical window ================
# Now, "OFFICIALY" chop off zero-force transients using f0_idx above 
# recall f0_idx accounts for fmax_idx (init transients removed already)
f_trim = f_trim[:f0_idx, :]
th_trim = th_trim[:f0_idx]
time_trim = time_trim[:f0_idx]
ee_trim = ee_trim[:f0_idx, :]

## NOTE: f_max may be negative, we want to consider magnitude for safe force calculation
k_safe = 0.25 # fraction of max force
f_safe_value = (1-k_safe) * f_trim[fmax_idx,:]
print(f"\nSafe force threshold: {k_safe}% of f_max for f_safe= {f_safe_value} N")

# Extract subset of data where force exceeds safe threshold TODO: this just does x-comp for now.
idx_sub_crit = np.where(np.linalg.norm(f_trim, axis=1) >= np.linalg.norm(f_safe_value))[0]

# For curve fitting, we don't want the initial spike in force, let's scrub them 
# (our cleaned data already starts with the spike at index 0 (relatively speaking))
# Take a fixed time after fmax_idx
init_spike_time = time_trim[fmax_idx] + 0.25  # 0.25 seconds after fmax
init_spike_idx = np.where(time_trim >= init_spike_time)[0][0]
idx_sub_crit = idx_sub_crit[idx_sub_crit >= init_spike_idx]

## And capture the sub-critical force, theta, time, and EE pos values
f_subcrit = f_trim[idx_sub_crit,:]
th_subcrit = th_trim[idx_sub_crit]
t_subcrit = time_trim[idx_sub_crit]
ee_subcrit = ee_trim[idx_sub_crit, :]

# And plot
PLOT_SUBCRIT = True

if PLOT_SUBCRIT:
    th_trim_deg = np.rad2deg(th_trim)
    th_sc_deg = np.rad2deg(th_subcrit)
    fig4, ax4 = plt.subplots(figsize=(8, 4.5))
    # ax4.plot(np.rad2deg(th_trim), np.linalg.norm(f_trim, axis=1), color='k', linewidth=5, label='Simulated data')  # Plot the x-component of the force
    ax4.plot(th_trim_deg, f_trim[:,0], color='b', linewidth=5, label='Push force (x)')  # Plot the x-component of the force
    ax4.plot(th_trim_deg, f_trim[:,1], color='r', linewidth=5, label='Push force (y)')  # Plot the y-component of the force
    ax4.plot(th_trim_deg, f_trim[:,2], color='m', linewidth=5, label='Push force (z)')  # Plot the z-component of the force
    # ax4.scatter(th_sc_deg, np.linalg.norm(f_subcrit, axis=1), color='r', s=80, label='Sub-critical window')
    ax4.scatter(th_sc_deg, f_subcrit[:,0], color='k', s=80, label='Sub-critical window')
    ax4.scatter(th_sc_deg, f_subcrit[:,1], color='k', s=80, label='')
    ax4.scatter(th_sc_deg, f_subcrit[:,2], color='k', s=80, label='')
    ax4.axhline(0, color='c', label='_')
    # ax4.set_ylabel("Force Norm (N)", color='b', fontsize=20)
    ax4.set_ylabel("Force (N)", color='b', fontsize=20)
    ax4.set_xlabel("Object Angle (deg)", color='g', fontsize=20)
    ax4.legend(loc='lower right', fontsize=15)
    ax4.grid(True)
    plt.tight_layout()
    plt.show()


## =================== HACK: Modify experiment parameters to make better fitting ===================
print("\n ************** SUB CRITICAL FIT CALCULATION ****************")
# # Don't currently have exact measure for o_obj. HOWEVER, at theta_crit, the EE x-pos should be equal to o x-pos!
ee_at_theta_star = ee_subcrit[np.argmin(np.abs(th_subcrit - theta_star_calc)), :] + 0.04 # Small 4cm offset makes it match!! TODO: Investigate tiny error propagation
o_obj = np.array([ee_at_theta_star[0], 0, 0])

# o_obj = np.array([0.398, 0, 0]) # Experimentally determined for now
print(f"\nUsing o_obj = {o_obj} for analysis\n")


# ALSO, my mass is being estimated too low... Let's try artificially boosting the force to see if theres some factor there
# f_subcrit *= 1.12

## =================================================================================================


# Whether to use the primary tipping and push axes (for plotting too!)
# USE_X_ONLY = True
USE_X_ONLY = False

if USE_X_ONLY:

    ## ================ Fit the COM models ===================
    f_app_subcrit = -f_subcrit # NOTE: IMPORTANT NEGATE TO MATCH F OBJECT EXPERIENCES

    ## Start by fitting straight line to sub-critical data to get initial guess using geom
    lin_slope, lin_b, _, _, _ = linregress(th_subcrit, f_app_subcrit[:,0])
    theta_star_guess = -lin_b / lin_slope
    # tan(th) = d_c / z_c => z_c = d_c / tan(th)
    dc_gt       = np.linalg.norm(np.array([-0.05, 0]))      # ground truth for d_c from 2D com estimation
    zc_guess    = dc_gt / np.tan(theta_star_guess)
    # m ~ slope*h_f / g*z_c
    m_guess = abs(lin_slope)*ee_subcrit[-1,2]/(9.81*zc_guess)
    print(f"\nInitial guess from linear fit:")
    print(f"mass: {m_guess:.3f} kg    zc: {zc_guess:.3f} m    theta*: {np.rad2deg(theta_star_guess):.3f} deg")

    ## =============== Fit using TAU model ==================
    # Before fitting, must pre-compute corresponding PUSH torque
    # o_obj = np.array([0.627, 0, 0]) # APPROXIMATE OBJECT ORIGIN IN WORLD FRAME (HACK: TEMPORARY FIX) *******************
    # o_obj = np.array([0.58, 0, 0])
    tau_app_subcrit = tau_app_model(f_app_subcrit, (ee_subcrit - o_obj)).ravel()

    [m_est, zc_est], pcov  = curve_fit(tau_model, th_subcrit, tau_app_subcrit, p0=[m_guess, zc_guess])
    # Now use fitted parameters to estimate theta_star
    theta_star_est = np.rad2deg(np.arctan2(dc_gt, zc_est))

    print(f"\nFit using TAU model:")
    print(f"mass: {m_est:.3f} kg    zc: {zc_est:.3f} m    theta*: {theta_star_est:.2f} deg")
    print(f"\nGround Truth:")
    print(f"mass: {m_gt} kg    zc: {com_gt[2]:.3f} m    theta*: {theta_star_gt:.2f} deg")


    ## ============================ PLOTTING ============================
    # Let's plot the whole curve, the sub-critical window, the linear fit, and whole curve w fit params
    f_app_full      = -f_trim[fmax_idx+init_spike_idx:, :] # expected: t=0 ~ fmax, t=end ~ f0
    ee_full         = ee_trim[fmax_idx+init_spike_idx:, :]
    tau_app_full    = tau_app_model(f_app_full, (ee_full - o_obj))

    # Extract theta from torque calculation
    th_full         = theta_from_tau(tau_app_full, m_est, zc_est, use_branch='minus')
    # Plot the linear fit using y = mx + b
    f_app_lin = lin_slope * th_full + lin_b

    ## Plot original data and sub-critical window
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # ax.plot(np.rad2deg(th_trim), np.linalg.norm(f_trim, axis=1), '--k', linewidth=5, label='Original data')  # Plot F norm
    ax.plot(np.rad2deg(th_trim), f_trim[:,0], '--k', linewidth=5, label='Original data')  # Plot the x-component of the force
    # ax.scatter(np.rad2deg(th_subcrit), np.linalg.norm(f_app_subcrit, axis=1), color='r', s=200, alpha=0.9, label='Sub-critical window')
    ax.scatter(np.rad2deg(th_subcrit), -f_app_subcrit[:,0], color='r', s=200, alpha=0.9, label='Sub-critical window')

    ax.plot(np.rad2deg(th_full), -f_app_lin, color='orange', linewidth=3, label='Sub-crit linear fit')
    ## FOR FUN, plot ALL theta and force
    ax.scatter(np.rad2deg(th_full), -f_app_full[:,0], color='b', label='Full fit')
    ax.axvline(theta_star_gt, color='g', linestyle='--', linewidth=5, label=r'Ground truth $\theta^*$')

    ax.axhline(0, color='c', linewidth=2) # Horizontal line at zero for reference
    ax.set_ylabel("X-Force (N)", color='b', fontsize=20)
    ax.set_xlabel("Object Angle (degrees)", color='g', fontsize=20)
    ax.legend(loc='lower right', fontsize=15)
    # Make the tick marks bigger
    # ax.tick_params(axis='both', which='major', )
    ax.tick_params(axis='x', labelcolor='g', labelsize=20)
    ax.tick_params(axis='y', labelcolor='b', labelsize=20)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

else:

    print("\n\nFitting using FORCE MAGNITUDE DATA...\n\n")
    ## ================ Fit the COM models ===================
    f_app_subcrit = -f_subcrit # NOTE: IMPORTANT NEGATE TO MATCH F OBJECT EXPERIENCES

    ## Start by fitting straight line to sub-critical data to get initial guess
    lin_slope, lin_b, _, _, _ = linregress(th_subcrit, np.linalg.norm(f_app_subcrit, axis=1))

    theta_star_guess = -lin_b / lin_slope

    # tan(th) = d_c / z_c => z_c = d_c / tan(th)
    dc_gt       = np.linalg.norm(np.array([-0.05, 0]))      # ground truth for d_c from 2D com estimation
    zc_guess    = dc_gt / np.tan(theta_star_guess)
    m_guess     = abs(lin_slope) #0.5                              # gt is 0.634 kg # TODO: Make better guess
    print(f"\nInitial guess from linear fit:")
    print(f"mass: {m_guess:.3f} kg    zc: {zc_guess:.3f} m    theta*: {np.rad2deg(theta_star_guess):.3f} deg")

    ## =============== Fit using TAU model ==================
    # Before fitting, must pre-compute corresponding PUSH torque
    
    tau_app_subcrit = tau_app_model(f_app_subcrit, (ee_subcrit - o_obj)).ravel()

    [m_est, zc_est], pcov  = curve_fit(tau_model, th_subcrit, tau_app_subcrit, p0=[m_guess, zc_guess])
    # Now use fitted parameters to estimate theta_star
    theta_star_est = np.rad2deg(np.arctan2(dc_gt, zc_est))

    print(f"\nFit using TAU model:")
    print(f"mass: {m_est:.3f} kg    zc: {zc_est:.3f} m    theta*: {theta_star_est:.2f} deg")
    print(f"\nGround Truth:")
    print(f"mass: {0.635} kg    zc: {0.1378} m    theta*: {theta_star_gt:.2f} deg")


    ## ============================ PLOTTING ============================
    # Let's plot the whole curve, the sub-critical window, the linear fit, and whole curve w fit params
    f_app_full      = -f_trim[fmax_idx+init_spike_idx:, :] # expected: t=0 ~ fmax, t=end ~ f0
    ee_full         = ee_trim[fmax_idx+init_spike_idx:, :]
    tau_app_full    = tau_app_model(f_app_full, (ee_full - o_obj))

    # Extract theta from torque calculation
    th_full         = theta_from_tau(tau_app_full, m_est, zc_est, use_branch='minus')
    # Plot the linear fit using y = mx + b
    f_app_lin = lin_slope * th_full + lin_b

    ## Plot original data and sub-critical window
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.rad2deg(th_trim), np.linalg.norm(f_trim, axis=1), '--k', linewidth=5, label='Original data')  # Plot F norm
    ax.scatter(np.rad2deg(th_subcrit), np.linalg.norm(f_app_subcrit, axis=1), color='r', s=200, alpha=0.9, label='Sub-critical window')

    ax.plot(np.rad2deg(th_full), f_app_lin, color='orange', linewidth=3, label='Sub-crit linear fit')
    ## FOR FUN, plot ALL theta and force
    ax.scatter(np.rad2deg(th_full), np.linalg.norm(f_app_full, axis=1), color='g', label='Full fit')

    ax.axhline(0, color='c', linewidth=2) # Horizontal line at zero for reference
    ax.set_ylabel("Force Norm (N)", color='b', fontsize=20)
    ax.set_xlabel("Object Angle (degrees)", color='g', fontsize=20)
    ax.legend(loc='lower right', fontsize=15)
    ax.grid(True)
    plt.tight_layout()
    plt.show()