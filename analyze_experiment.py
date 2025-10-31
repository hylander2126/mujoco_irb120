import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt
from scipy.interpolate import interp1d
from scripts.utils.com_estimation import theta_model, align_zeros
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
path = "experiments/run_2025-10-16_12-04-17_t001_ft.csv"


## Extract force
f_data = read_csv(path, trim_rows=1)
f_exp_raw = np.zeros((len(f_data), 3))                      # n_samples x 3 (fx, fy, fz)
for i, row in enumerate(f_data):
    f_exp_raw[i, :] = [float(row[j + 1]) for j in range(3)]  # fx, fy, fz are in columns 1, 2, 3

## Extract angle(s)
tag_data = read_csv(path.replace("ft", "tag"), trim_rows=1) # Discard headers
tag_exp_raw = np.zeros((len(tag_data), 3))                      # n_samples x 3 (roll, pitch, yaw)
for i, row in enumerate(tag_data):
    tag_exp_raw[i, :] = [float(row[j + 2]) for j in range(3)]  # roll, pitch, yaw are in columns 2, 3, 4

## Extract time (from both f and th in order to downsample later)
time_f = np.array([float(row[0]) for row in f_data])     # 'force' time is in column 0
time_f -= time_f[0]                                       # Normalize time to start at zero
time = np.array([float(row[0]) for row in tag_data])     # time is in column 0 (let's use tag_data since its already the correct length)
time -= time[0]    

## ================ Process the data (specifically force) ===================
# Butterworth filter
b, a = butter(4, 5, fs=500, btype='low') # order, cutoff frequency, fs
f_exp_filt = filtfilt(b, a, f_exp_raw, axis=0)

# And interpolate to match the time steps of the angle data
f_exp_interp = np.array([np.interp(time, time_f, f_exp_filt[:, i]) for i in range(3)]).T

# Let's bias the force and angle data using START & END values (end looks better for f, start for th)
f_exp_interp -= np.mean(f_exp_interp[-20:, :], axis=0)
tag_exp_raw -= np.mean(tag_exp_raw[:10], axis=0)


th_exp = tag_exp_raw[:, 2]  # HACK: assume ONLY tip in 'yaw' axis (column 2)


# ================ Find contact, settling, and start moments ===================
# First, let's find when we make contact (will be the maximal magnitude of force)
contact_idx_orig = np.argmax(np.linalg.norm(f_exp_interp, axis=1))
contact_time_orig = time[contact_idx_orig]

# Then, let's find when the angle settles (after contact)
settle_indices_orig = np.where(np.isclose(th_exp, th_exp[-1], atol=1e-2))[0]#[0]
valid_settle_indices = settle_indices_orig[settle_indices_orig > contact_idx_orig]          # make sure we only look for settling AFTER contact
if len(valid_settle_indices) == 0:
    print("Warning: No settling found after contact!")
    settle_idx_orig = len(time) - 1 # fall back to end of data
else:
    settle_idx_orig = valid_settle_indices[0]  # take the first one
settle_time_orig = time[settle_idx_orig]

# Define start of analysis window (a few seconds before contact)
start_time_orig = contact_time_orig - 0.25
start_idx_orig = np.where(time >= start_time_orig)[0][0]

print(f"Contact detected at index {contact_idx_orig} ({contact_time_orig:.2f} s)")
print(f"Settling detected at index {settle_idx_orig} ({settle_time_orig:.2f} s)")
print(f"Analysis window will start at index {start_idx_orig} ({time[start_idx_orig]:.2f} s)")


# =============== Trim the data to the analysis window ===================
end_idx_orig = settle_idx_orig + 5  # add a few extra samples to be safe

time_trim = np.array(time[start_idx_orig:end_idx_orig])
f_trim = np.array(f_exp_interp[start_idx_orig:end_idx_orig, :])
th_trim = np.array(th_exp[start_idx_orig:end_idx_orig])
print(f"Rough time range of contact window: {time_trim[0]:.2f} s to {time_trim[-1]:.2f} s\n")

# =============== Analysis on trimmed data ===================
fmax_idx = np.argmax(np.linalg.norm(f_trim, axis=1))
print(f"fmax time: {time_trim[fmax_idx]:.2f} s, fmax force: {f_trim[fmax_idx, :]}, fmax angle: {np.rad2deg(th_trim[fmax_idx]):.2f} degrees")

# For clarity, let's define data *AFTER* max force event
f_contact = f_trim[fmax_idx:, :]
th_contact = th_trim[fmax_idx:]
time_contact = time_trim[fmax_idx:]

# Determine near-zero-crossing of force in primary axis (x) by searching after max force (fmax_idx) index
f_is_near_zero = np.isclose(f_contact[:, 0], 0, atol=1e-2)

# Select angles and times at those near-zero-crossings
th_departure = th_contact[f_is_near_zero]
time_departure = time_contact[f_is_near_zero]
print(f"Corresponding angles and times at zero crossings: {np.rad2deg(th_departure)}, {time_departure}")

# Get average of those first few near-zero-crossings
if len(th_departure) == 0:
    print("Warning: No zero crossings found after max force!")
    avg_th_departure = np.nan
    avg_time_departure = np.nan
elif len(th_departure) < 3:
    print("Warning: Fewer than 3 zero crossings found after max force, averaging what we have.")
    avg_th_departure = np.mean(th_departure)
    avg_time_departure = np.mean(time_departure)
else:
    avg_th_departure = np.mean(th_departure[:3])  # average first 3 crossings
    avg_time_departure = np.mean(time_departure[:3])  # average first 3 crossings

print(f"Mean time and angle at initial zero crossings: {avg_time_departure:.2f} s, {np.rad2deg(avg_th_departure):.2f} degrees")

# And from simulation, we know the ground truth
theta_star_gt = np.rad2deg(np.arctan2(0.05, 0.15))
print(f"theta_star (ground truth) = {theta_star_gt:.2f} degrees")


# ================ Plot the data ===================
PLOT_RAW = False
PLOT_XYZ = True
PLOT_RELATIONSHIP = False

# Now let's plot the x_data over time
if PLOT_RAW:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax2 = plt.twinx()
    ax.plot(time_f, f_exp_raw[:, 0], "b", label='X Force (raw)')
    ax.plot(time_f, f_exp_raw[:, 1], "r", label='Y Force (raw)')
    ax.plot(time_f, f_exp_raw[:, 2], "m", label='Z Force (raw)')
    ax2.plot(time, tag_exp_raw[:, 0], color='g', linestyle='-', label='Roll (raw)')
    ax2.plot(time, tag_exp_raw[:, 1], color='c', linestyle='-', label='Pitch (raw)')
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

# Then determine first zero-crossing of x-component of force (after initial transient)
f0_idx = fmax_idx + np.where(np.isclose(f_trim[fmax_idx:,0], 0, atol=2e-3))[0][0]

# Now, "OFFICIALY" chop off zero-force transients using f0_idx above
f_trim = f_trim[:f0_idx, :]
th_trim = th_trim[:f0_idx]
time_trim = time_trim[:f0_idx]

K_SAFE = 0.25 # fraction of max force
f_safe_value = (1-K_SAFE) * f_trim[fmax_idx,:]
print(f"Safe force threshold: {K_SAFE}% of f_max for f_safe= {f_safe_value} N")

# Extract subset of data where force exceeds safe threshold TODO: this just does x-comp for now.
idx_sub_crit = np.where(np.linalg.norm(f_trim, axis=1) >= np.linalg.norm(f_safe_value))[0]

# For curve fitting, we don't want the initial spike in force, let's scrub them (our cleaned data already starts with the spike at index 0 (relatively speaking))
init_spike_idx = 8 #4 # Manually determined for now
idx_sub_crit = idx_sub_crit[init_spike_idx:]  # Keep indices 80 onward

## And record the sub-critical force, theta, and time values
f_app_subcrit = -f_trim[idx_sub_crit,:] # NOTE: IMPORTANT NEGATE TO MATCH F OBJECT EXPERIENCES
th_subcrit = th_trim[idx_sub_crit]
t_subcrit = time_trim[idx_sub_crit]


PLOT_SUBCRIT = False

if PLOT_SUBCRIT:
    fig4, ax4 = plt.subplots(figsize=(8, 4.5))
    ax4.plot(np.rad2deg(th_trim), np.linalg.norm(f_trim, axis=1), color='k', linewidth=5, label='Simulated data')  # Plot the x-component of the force
    ax4.scatter(np.rad2deg(th_subcrit), np.linalg.norm(f_app_subcrit, axis=1), color='r', s=80, label='Sub-critical window')
    ax4.axhline(0, color='c', label='_')
    ax4.set_ylabel("Force Norm (N)", color='b', fontsize=20)
    ax4.set_xlabel("Object Angle (deg)", color='g', fontsize=20)
    ax4.legend(loc='upper right', fontsize=15)
    ax4.grid(True)
    plt.show()


## ================ Fit the COM models ===================

## Start by fitting straight line to sub-critical data to get initial guess

# Don't want norm of force, want primary force direction, let's determine the primary direction, THIS MIGHT NOT BE GLOBAL AXIS ALIGNED
# x_intercepts = np.zeros((3,))
# for i in range(3):
#     slope, y_int, _, _, _ = linregress(th_subcrit, f_app_subcrit[:,i])
#     x_intercepts[i] = -y_int / slope

# OR just use the force norm for linear fit...
lin_slope, lin_b, _, _, _ = linregress(th_subcrit, np.linalg.norm(f_app_subcrit, axis=1))

# Check out all theta guesses and pick the one with reasonable value (not close to zero!)
# theta_star_guess = x_intercepts[np.where(x_intercepts > 1e-2)[0][0]]
theta_star_guess = -lin_b / lin_slope

# tan(th) = d_c / z_c => z_c = d_c / tan(th)
d_c         = np.linalg.norm(np.array([-0.05, 0]))      # ground truth for d_c from 2D com estimation
zc_guess    = d_c / np.tan(theta_star_guess) # gt is 0.15 m
m_guess     = 0                              # gt is 0.634 kg
print(f"Initial guess from linear fit: theta*: {np.rad2deg(theta_star_guess):.3f} deg    mass: {m_guess:.3f} kg    zc: {zc_guess:.3f} m")
    
## Fit using THETA model
popt, pcov = curve_fit(theta_model, f_app_subcrit, th_subcrit, p0=[m_guess, zc_guess])
m_est_th, zc_est_th = popt
print(f"\nFit using THETA model: mass: {m_est_th:.3f} kg    zc: {zc_est_th:.3f} m")
# Use fitted parameters to generate estimated THETA values over sub-critical FORCE data
th_pred_fit = theta_model(f_app_subcrit, m_est_th, zc_est_th)

## Back-check by estimating force zero-crossing (or theta_star_est)
theta_star_est_th = np.rad2deg(np.arctan2(0.05, zc_est_th))
print(f"Estimated theta* (THETA model): {theta_star_est_th:.2f} degrees")
print(f"Ground Truth: {theta_star_gt:.2f} degrees")

# And let's compare against all recorded force for fun (and visualization!)
f_all = f_trim[fmax_idx+init_spike_idx:, :]
th_all = theta_model(f_all, m_est_th, zc_est_th)  # use all measured forces to get theta estimates
t_all = time_trim[fmax_idx+init_spike_idx:]

## Plot original data and sub-critical window
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(np.rad2deg(th_trim), np.linalg.norm(f_trim, axis=1), color='k', linewidth=5, label='Original data')  # Plot the x-component of the force
ax.scatter(np.rad2deg(th_subcrit), np.linalg.norm(f_app_subcrit, axis=1), color='r', s=100, alpha=0.9, label='Sub-critical window')

## Plot fitted THETA curve
# ax.plot(np.rad2deg(th_pred_fit), np.linalg.norm(f_sub_crit, axis=1), color='k', linewidth=2, linestyle='--', label='Fitted curve (THETA model)')

## FOR FUN, plot ALL theta and force
ax.scatter(np.rad2deg(th_all), np.linalg.norm(f_all, axis=1), color='g', label='Full fit')

ax.axhline(0, color='c', linewidth=2) # Horizontal line at zero for reference
ax.set_ylabel("Force Norm (N)", color='b', fontsize=20)
ax.set_xlabel("Object Angle (degrees)", color='g', fontsize=20)
ax.legend(loc='upper right', fontsize=15)
ax.grid(True)
plt.tight_layout()
plt.show()