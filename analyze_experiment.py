import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt
from scipy.interpolate import interp1d
from scripts.utils.com_estimation import tau_app_model, tau_model, theta_from_tau, align_zeros
from scipy.stats import linregress
from scipy.optimize import curve_fit


def read_csv(file_path, trim_rows=0):
    with open(file_path, mode ='r')as file:
        csvFile = csv.reader(file)
        csv_arr = []
        for lines in csvFile:
                csv_arr.append(lines)
        return csv_arr[trim_rows:]

# ================ Load the data ===================
# path = "../experiments/20251006_1.csv"
# path = "../experiments/run_2025-10-10_17-04-01_t001_ft.csv"
# path = "../experiments/run_2025-10-15_15-44-01_t001_ft.csv"
# path = "experiments/run_2025-10-15_17-32-23_t001_ft.csv"
# path = "experiments/run_2025-10-16_12-04-17_t001_ft.csv"
# path_root = "experiments/run_2025-11-03_11-59-03_t001_"
path = "experiments/run_2025-11-03_17-57-03_t001_SYNC_FTCLK.csv"
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

# Quick make sure we are using correct columns
print(f"Data loaded from {path}:")
print(f"Initial F: {f_exp_raw[0, :]}, \nTag orientation: {tag_exp_raw[0, :]}, \nEE pos: {ee_exp[0, :]}\n")

## ================ Process the data (specifically force) ===================
# Butterworth filter
b, a         = butter(2, 4, fs=500, btype='low') # 4,5,500 order, cutoff frequency, fs
f_exp_filt   = filtfilt(b, a, f_exp_raw, axis=0)



# TEMPORARY *************************************** TEMPORARY ***************
# Before filtering tag, get rid of nans
nan_indices = np.isnan(tag_exp_raw).any(axis=1)
if np.any(nan_indices):
    print(f"Warning: Found {np.sum(nan_indices)} NaN entries in tag data, performing interpolation to fill.")
    valid_indices = ~nan_indices
    for col in range(tag_exp_raw.shape[1]):
        interp_func = interp1d(time[valid_indices], tag_exp_raw[valid_indices, col], kind='linear', fill_value="extrapolate")
        tag_exp_raw[nan_indices, col] = interp_func(time[nan_indices])
tag_exp_filt = filtfilt(b, a, tag_exp_raw, axis=0)
# ************************************************ TEMPORARY ***************

# We will have to go and set the 'clock' to the slowest frequency source later... This will probably be the tag detections



# Let's bias the force and angle data using START & END values (end looks better for f, start for th)
f_exp_filt -= np.mean(f_exp_filt[-20:, :], axis=0)
tag_exp_filt -= np.mean(tag_exp_filt[:10], axis=0)

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
print(f"Analysis window will start at index {start_idx_og} ({time[start_idx_og]:.2f} s)")

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
print(f"fmax time: {time_trim[fmax_idx]:.2f} s, fmax force: {f_trim[fmax_idx, :]}, fmax angle: {np.rad2deg(th_trim[fmax_idx]):.2f} degrees")

# For clarity, let's only consider data when in contact (*AFTER* max force event)
# f_contact = f_trim[fmax_idx:, :]
# th_contact = th_trim[fmax_idx:]
# ee_contact = ee_trim[fmax_idx:, :]
# time_contact = time_trim[fmax_idx:]

# Determine near-zero-crossing of force in primary axis (x) by searching after max force (fmax_idx) index
f0_idxs = np.where(np.isclose(f_trim[:, 0], 0, atol=1e-2))[0]

f0_idx = f0_idxs[-1]  # take last crossing before end of data
print(f"f0 (last found) idx: {f0_idx}, num crossings found: {len(f0_idxs)}")

# # Select angles and times at those near-zero-crossings
# th_departure = th_contact[f0_idxs]
# time_departure = time_contact[f0_idxs]
# print(f"Corresponding angles and times at zero crossings: {np.rad2deg(th_departure)}, {time_departure}")

# # Get average of those first few near-zero-crossings
# if len(th_departure) == 0:
#     print("Warning: No zero crossings found after max force!")
#     avg_th_departure = np.nan
#     avg_time_departure = np.nan
# elif len(th_departure) < 3:
#     print("Warning: Fewer than 3 zero crossings found after max force, averaging what we have.")
#     avg_th_departure = np.mean(th_departure)
#     avg_time_departure = np.mean(time_departure)
# else:
#     avg_th_departure = np.mean(th_departure[:3])  # average first 3 crossings
#     avg_time_departure = np.mean(time_departure[:3])  # average first 3 crossings

# print(f"Mean time and angle at initial zero crossings: {avg_time_departure:.2f} s, {np.rad2deg(avg_th_departure):.2f} degrees")


## ==================== CALCULATE THETA* AND ZC ====================
theta_star_calc = th_trim[f0_idx]
print(f"\nCalculated theta* = {np.rad2deg(theta_star_calc):.2f} degrees")

xc_gt = 0.05
zc_calc = xc_gt / np.tan(theta_star_calc)
print(f"Calculated zc = {zc_calc:.3f} m")

# ground truth from sim
theta_star_gt = np.rad2deg(np.arctan2(0.05, 0.15))
print(f"\nInverting the equation, \nGround truth theta*: {theta_star_gt:.2f} degrees")
print(f'Ground truth zc = 0.15 m\n')


# ================ Plot the data ===================
PLOT_RAW = True
PLOT_XYZ = True
PLOT_RELATIONSHIP = True

# Now let's plot the x_data over time
if PLOT_RAW:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax2 = plt.twinx()
    ax.plot(time, f_exp_raw[:, 0], "b", label='X Force (raw)')
    ax.plot(time, f_exp_raw[:, 1], "r", label='Y Force (raw)')
    ax.plot(time, f_exp_raw[:, 2], "m", label='Z Force (raw)')
    # ax2.plot(time_th, tag_exp_raw[:, 0], color='g', linestyle='-', label='Roll (raw)')
    # ax2.plot(time_th, tag_exp_raw[:, 1], color='c', linestyle='-', label='Pitch (raw)')
    ax2.plot(time, tag_exp_raw[:, 2], color='y', linestyle='-', label='Yaw (raw)')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='g')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Force (N)', color='b', fontsize=10)
    ax.set_title('Raw X, Y, Z Data Over Time')
    ax.legend()
    ax.grid(True)
    align_zeros([ax, ax2])
    plt.show()

if PLOT_XYZ:
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = plt.twinx()
    ax1.plot(time_trim, f_trim[:, 0], "b", linewidth=5, label='Push force (x)')
    ax1.plot(time_trim, f_trim[:, 1], "r", linewidth=5, label='Push force (y)')
    ax1.plot(time_trim, f_trim[:, 2], "m", linewidth=5, label='Push force (z)')
    ax2.plot(time_trim, np.rad2deg(th_trim), color='g', linewidth=5, linestyle='-', label='Object angle')
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelcolor='b', labelsize=20)
    ax2.tick_params(axis='y', labelcolor='g', labelsize=20)
    ax2.set_ylabel('Object Angle (degrees)', color='g', fontsize=20)
    # ax2.set_ylim(-5, 90)
    ax1.set_xlabel('Time (s)', fontsize=20)
    ax1.set_ylabel('Force (N)', color='b', fontsize=20)
    # ax1.set_title('X, Y, Z Data Over Time')
    fig.legend(loc='upper left', bbox_to_anchor=(0.2, 0.9), fontsize=13)
    ax1.grid(True)
    align_zeros([ax1, ax2])
    plt.tight_layout()
    plt.show()

if PLOT_RELATIONSHIP:
    # NOTE: This is plotting the primary pushing force, NOT the magnitude.
    fig, ax3 = plt.subplots(figsize=(9, 4.5))
    # ax3.plot(np.rad2deg(th_trim), np.linalg.norm(f_exp_filt, axis=1), "k", label='Force Magnitude')
    ax3.plot(np.rad2deg(th_trim), f_trim[:, 0], "b", alpha=0.25, label='X-Force (raw)')
    ax3.axhline(0, color='c', linewidth=2)
    ax3.set_xlabel('Object Angle (degrees)', color='g', fontsize=20)
    ax3.set_ylabel('X-Force Magnitude (N)', color='b', fontsize=20)
    ax3.set_title('Primary Axis (X) Force vs. Object Angle')
    # ax3.set_xlim([-1, 25])
    # ax3.set_ylim([-0.05, fmax+0.2])
    ax3.axvline(theta_star_gt, color='g', linestyle='--', linewidth=2, label=r'Ground Truth $\theta_*$')
    ax3.legend()
    ax3.grid(True)
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
print(f"Safe force threshold: {k_safe}% of f_max for f_safe= {f_safe_value} N")

# Extract subset of data where force exceeds safe threshold TODO: this just does x-comp for now.
idx_sub_crit = np.where(np.linalg.norm(f_trim, axis=1) >= np.linalg.norm(f_safe_value))[0]

# For curve fitting, we don't want the initial spike in force, let's scrub them (our cleaned data already starts with the spike at index 0 (relatively speaking))
init_spike_idx = 8 #4 # Manually determined for now
idx_sub_crit = idx_sub_crit[init_spike_idx:]  # Keep indices 80 onward

## And capture the sub-critical force, theta, time, and EE pos values
f_subcrit = f_trim[idx_sub_crit,:]
th_subcrit = th_trim[idx_sub_crit]
t_subcrit = time_trim[idx_sub_crit]
ee_subcrit = ee_trim[idx_sub_crit, :]

# And plot
PLOT_SUBCRIT = False

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
    ax4.legend(loc='upper right', fontsize=15)
    ax4.grid(True)
    plt.show()


## ================ Fit the COM models ===================
f_app_subcrit = -f_subcrit # NOTE: IMPORTANT NEGATE TO MATCH F OBJECT EXPERIENCES

## Start by fitting straight line to sub-critical data to get initial guess
# lin_slope, lin_b, _, _, _ = linregress(th_subcrit, np.linalg.norm(f_app_subcrit, axis=1))
lin_slope, lin_b, _, _, _ = linregress(th_subcrit, f_app_subcrit[:,0])

theta_star_guess = -lin_b / lin_slope

# tan(th) = d_c / z_c => z_c = d_c / tan(th)
dc_gt       = np.linalg.norm(np.array([-0.05, 0]))      # ground truth for d_c from 2D com estimation
zc_guess    = dc_gt / np.tan(th_subcrit[-1])
m_guess     = abs(lin_slope) #0.5                              # gt is 0.634 kg # TODO: Make better guess
print(f"Initial guess from linear fit:")
print(f"mass: {m_guess:.3f} kg    zc: {zc_guess:.3f} m    theta*: {np.rad2deg(theta_star_guess):.3f} deg")

## =============== Fit using TAU model ==================
# Before fitting, must pre-compute corresponding PUSH torque
o_obj = np.array([0.627, 0, 0]) # APPROXIMATE OBJECT ORIGIN IN WORLD FRAME (HACK: TEMPORARY FIX) *******************
tau_app_subcrit = tau_app_model(f_app_subcrit, (ee_subcrit - o_obj)).ravel()

com_gt = [0, 0, 0.15] # TEMP because ground truth is saved to sim obj model...

[m_est, zc_est], pcov  = curve_fit(tau_model, th_subcrit, tau_app_subcrit, p0=[m_guess, zc_guess])
# Now use fitted parameters to estimate theta_star
theta_star_est = np.rad2deg(np.arctan2(dc_gt, zc_est))

print(f"\nFit using TAU model:")
print(f"mass: {m_est:.3f} kg    zc: {zc_est:.3f} m    theta*: {theta_star_est:.2f} deg")
print(f"\nGround Truth:")
print(f"mass: {0.635} kg    zc: {com_gt[2]:.3f} m    theta*: {theta_star_gt:.2f} deg")


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
# ax.scatter(np.rad2deg(th_full), np.linalg.norm(f_app_full, axis=1), color='g', label='Full fit')
ax.scatter(np.rad2deg(th_full), -f_app_full[:,0], color='g', label='Full fit')

ax.axhline(0, color='c', linewidth=2) # Horizontal line at zero for reference
ax.set_ylabel("Force Norm (N)", color='b', fontsize=20)
ax.set_xlabel("Object Angle (degrees)", color='g', fontsize=20)
ax.legend(loc='upper right', fontsize=15)
ax.grid(True)
plt.tight_layout()
plt.show()