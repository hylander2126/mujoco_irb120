import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import utils.com_estimation as com


def read_csv(file_path, trim_rows=0):
    with open(file_path, mode ='r')as file:
        csvFile = csv.reader(file)
        csv_arr = []
        for lines in csvFile:
                csv_arr.append(lines)
        return csv_arr[trim_rows:]

# path = "../experiments/20251006_1.csv"
path = "../experiments/run_2025-10-10_17-04-01_t001_ft.csv"
f_data = read_csv(path, trim_rows=1)
th_data = read_csv(path.replace("ft", "tag"), trim_rows=1)

# Extract the relevant columns 
f_exp = np.zeros((len(f_data), 3))                      # n_samples x 3 (fx, fy, fz)
for i, row in enumerate(f_data):
    for j in range(3):
        f_exp[i, j] = float(row[j + 1])                 # fx, fy, fz are in columns 1, 2, 3

# TEMP HACK: just assume we ONLY tip in 'yaw' axis
th_exp = np.array([float(row[4]) for row in th_data])   # theta is in column 4

time = np.array([float(row[0]) for row in th_data])     # time is in column 0
time -= time[0]                                         # Normalize time to start at zero

## ================ Process the data (specifically force) ===================

f_exp_filt = medfilt(f_exp, kernel_size=(111,1)) # Reduce noise
# Unfortunately, our pre-processing (correction of axes) didn't work or I got confused. 
arr_holder = f_exp_filt.copy()
f_exp_filt[:, 0] = arr_holder[:, 2]   # X : Decreases rapidly (goes negative) from zero and then slowly increases back to zero
f_exp_filt[:, 1] = arr_holder[:, 0]   # Y : Should hover around zero
f_exp_filt[:, 2] = -arr_holder[:, 1]  # Z : Slowly ramps up from zero then back down to zero
# For some reason the x data is showing no gravity compensation
# Let's take the average reading from the first 100 samples and subtract it from the x data
# x_offset = np.mean(f_exp_filt[:100, 0])
# f_exp_filt[:, 0] -= -0.26*np.ones_like(f_exp_filt[:, 0]) # x_offset

# Let's make sure theta is same length as f_exp_filt
idxs = np.linspace(0, len(f_exp_filt)-1, len(th_exp), dtype=int)
f_exp_filt = f_exp_filt[idxs, :]


# === Calculate theta_star ===

# Find index of f_max (after initial transient)
fmax = np.max(np.linalg.norm(f_exp_filt, axis=1))
fmax_idx = f_exp_filt[:, 0].argmax()
# Then determine first zero-crossing of x-component of force (after initial transient)
# fzero_idx = fmax_idx + np.where(np.isclose(f_exp_filt[fmax_idx:, 0], 0, atol=1e-3))[0][0]

theta_star_calc = th_exp[fmax_idx]
print(f"theta_star (calculated) = {np.rad2deg(theta_star_calc):.2f} degrees")

# And from simulation, we know the ground truth
theta_star_gt = np.rad2deg(np.arctan2(0.05, 0.15))
print(f"theta_star (ground truth) = {theta_star_gt:.2f} degrees")

f_exp_offset = f_exp_filt[:, 0].copy()
f_exp_offset -= -0.26*np.ones_like(f_exp_filt[:, 0])


# ================ Plot the data ===================
PLOT_INDIVIDUALS = True
PLOT_RELATIONSHIP = True

# Now let's plot the x_data over time
if PLOT_INDIVIDUALS:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = plt.twinx()
    ax1.plot(time, f_exp_filt[:, 0], "b", label='X Force (raw)')
    ax1.plot(time, f_exp_filt[:, 1], "r", label='Y Force (raw)')
    ax1.plot(time, f_exp_filt[:, 2], "m", label='Z Force (raw)')
    ax2.plot(time, np.rad2deg(th_exp), color='g', linestyle='-', label='Payload pitch')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylabel('Payload Pitch (degrees)', color='g', fontsize=10)
    ax2.set_ylim(-5, 30)
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Force (N)', color='b', fontsize=10)
    ax1.set_title('X, Y, Z Data Over Time')
    ax1.legend()
    ax1.grid(True)
    com.align_zeros([ax1, ax2])
    plt.show()

if PLOT_RELATIONSHIP:
    fig, ax3 = plt.subplots(figsize=(10, 6))
    # ax3.plot(np.rad2deg(th_exp), np.linalg.norm(f_exp_filt, axis=1), "k", label='Force Magnitude')
    ax3.plot(np.rad2deg(th_exp), abs(f_exp_filt[:, 0]), "b", alpha=0.25, label='X-Force (raw)')
    ax3.plot(np.rad2deg(th_exp), abs(f_exp_offset), color="k", label='X-Force (offset)')
    ax3.axhline(0, color='c', linewidth=2)
    ax3.set_xlabel('Payload Pitch (degrees)', color='g', fontsize=10)
    ax3.set_ylabel('X-Force Magnitude (N)', color='b', fontsize=10)
    ax3.set_title('Force vs. Payload Pitch')
    ax3.set_xlim([-1, 25])
    ax3.set_ylim([-0.05, fmax+0.2])
    ax3.axvline(theta_star_gt, color='g', linestyle='--', linewidth=2, label=r'Ground Truth $\theta_*$')
    ax3.legend()
    ax3.grid(True)
    plt.show()
