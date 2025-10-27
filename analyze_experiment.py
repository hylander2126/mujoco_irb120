import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt
from scipy.interpolate import interp1d
import scripts.utils.com_estimation as com


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
start_time_orig = contact_time_orig - 1.0
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
PLOT_INDIVIDUALS = False
PLOT_RELATIONSHIP = True

# Now let's plot the x_data over time
if PLOT_RAW:
    fig, ax = plt.subplots(figsize=(10, 6))
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
    com.align_zeros([ax, ax2])
    plt.show()

if PLOT_INDIVIDUALS:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = plt.twinx()
    ax1.plot(time_trim, f_trim[:, 0], "b", label='X Force')
    ax1.plot(time_trim, f_trim[:, 1], "r", label='Y Force')
    ax1.plot(time_trim, f_trim[:, 2], "m", label='Z Force')
    ax2.plot(time_trim, np.rad2deg(th_trim), color='g', linestyle='-', label='Payload Angle')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylabel('Payload Angle (degrees)', color='g', fontsize=10)
    ax2.set_ylim(-5, 50)
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Force (N)', color='b', fontsize=10)
    ax1.set_title('X, Y, Z Data Over Time')
    fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
    ax1.grid(True)
    com.align_zeros([ax1, ax2])
    plt.show()

if PLOT_RELATIONSHIP:
    fig, ax3 = plt.subplots(figsize=(10, 6))
    # ax3.plot(np.rad2deg(th_trim), np.linalg.norm(f_exp_filt, axis=1), "k", label='Force Magnitude')
    ax3.plot(np.rad2deg(th_trim), f_trim[:, 0], "b", alpha=0.25, label='X-Force (raw)')
    ax3.axhline(0, color='c', linewidth=2)
    ax3.set_xlabel('Payload Angle (degrees)', color='g', fontsize=10)
    ax3.set_ylabel('X-Force Magnitude (N)', color='b', fontsize=10)
    ax3.set_title('Primary Axis (X) Force vs. Payload Angle')
    # ax3.set_xlim([-1, 25])
    # ax3.set_ylim([-0.05, fmax+0.2])
    ax3.axvline(theta_star_gt, color='g', linestyle='--', linewidth=2, label=r'Ground Truth $\theta_*$')
    ax3.legend()
    ax3.grid(True)
    plt.show()


## Now let's extract the sub-critical window data for model fitting
## NOTE: f_max may be negative, we want to consider magnitude for safe force calculation
k_safe = 0.25 # fraction of max force
f_safe_value = (1-k_safe) * f_trim[fmax_idx,:]
print(f"Safe force threshold: {k_safe}% of f_max for f_safe= {f_safe_value} N")

# Extract subset of data where force exceeds safe threshold TODO: this just does x-comp for now.
# idx_sub_crit = np.where(abs(f_hist_filt[:,0]) >= abs(f_safe[0]))[0]
idx_sub_crit = np.where(np.linalg.norm(f_trim, axis=1) >= np.linalg.norm(f_safe_value))[0]

# For curve fitting, we don't want the initial spike in force, let's scrub them (our cleaned data already starts with the spike at index 0 (relatively speaking))
init_spike_idx = 4 #160 # Manually determined for now
idx_sub_crit = idx_sub_crit[init_spike_idx:]  # Keep indices 80 onward

## And record the sub-critical force, theta, and time values
f_sub_crit = f_trim[idx_sub_crit,:]
th_sub_crit = th_trim[idx_sub_crit]
t_sub_crit = time_trim[idx_sub_crit]

# print(f"Sub-critical window has {len(th_sub_crit)} samples from time {t_sub_crit[0]:.2f} s to {t_sub_crit[-1]:.2f} s")

# And plot
fig4, ax4 = plt.subplots(figsize=(8, 4.5))
ax4.plot(np.rad2deg(th_trim), np.linalg.norm(f_trim, axis=1), color='k', linewidth=5, label='Simulated data')  # Plot the x-component of the force
ax4.scatter(np.rad2deg(th_sub_crit), np.linalg.norm(f_sub_crit, axis=1), color='r', s=80, label='Sub-critical window')
ax4.axhline(0, color='c', label='_')
ax4.set_ylabel("Force Norm (N)", color='b', fontsize=20)
ax4.set_xlabel("Object Angle (deg)", color='g', fontsize=20)
ax4.legend(loc='upper right', fontsize=15)
ax4.grid(True)
plt.show()
