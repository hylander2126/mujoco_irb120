import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt, savgol_filter
from scipy.interpolate import interp1d
from scripts.utils.com_estimation import tau_app_model, tau_model, align_zeros, F_model
from scipy.stats import linregress
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

OBJECTS = {
        # X and Y is distance from tipping edge (object frame!) to projected CoM on table plane
        "box": {
            # "path": "experiments/20251208_155739_box_t02.csv",
            "path": "experiments/20251215_170303_box_t01.csv",
            "com": [-0.0500, 0.0, 0.1500],
            "mass": 0.635,
            "height": 0.3, # 300 mm
            "est": [0,0,0],
        },
        "heart": {
            "path": "experiments/20251208_155739_heart_t03.csv",
            "com": [-0.0458, 0, 0.0800], # [-0.0458, 0, 0.1]
            "mass": 0.269,
            "height": 0.2, # 200 mm
            "est": [0,0,0],
        },
        "flashlight": {
            # "path": "experiments/20251208_143007_flashlight_t01.csv",  # older test
            "path": "experiments/20251208_155739_flashlight_t04.csv",    # new force stop test
            "com": [-0.0250, 0.0, 0.0950],
            "mass": 0.386,
            "height": 0.2, # 200 mm
            "est": [0,0,0],
        },
        "lshape": {
            "path": "experiments/20251208_155739_lshape_t05.csv",
            "com": [-0.0250, 0.0, 0.0887],
            "mass": 0.118,
            "height": 0.15, # 150 mm
            "est": [0,0,0],
        },
        "monitor": {
            "path": "experiments/20251210_182429_monitor_t03.csv",
            "com": [-0.1118, 0.0, 0.2362],
            "mass": 5.37,
            "height": 0.515, # 515 mm
            "est": [0,0,0],
        }
    }

def read_csv(file_path, trim_rows=0):
    with open(file_path, mode ='r')as file:
        csvFile = csv.reader(file)
        csv_arr = []
        for lines in csvFile:
                csv_arr.append(lines)
        return csv_arr[trim_rows:]


def main(shape, csv_path, com_gt, m_gt, theta_star_gt, plot_raw=True, plot_ee=False):

    csv_data = read_csv(csv_path, trim_rows=1)  # Discard headers

    # TODO: Figure out the correct NEW columns now that we have quaternion for tag data

    ## ================ Extract time series data ===================
    time        = np.zeros(len(csv_data))
    ee_exp      = np.zeros((len(csv_data), 3))
    f_exp_raw   = np.zeros((len(csv_data), 3)) # RAW because we are filtering later
    tag_exp_raw = np.zeros((len(csv_data), 3)) # RAW because we are filtering later
    cnt_exp     = np.zeros(len(csv_data))      # Contact boolean
    trig_exp    = np.zeros(len(csv_data))      # Trigger boolean

    # Column nums for each data type: (time is column zero)
    f_cols      = [1, 2, 3]
    ee_cols     = [7, 8, 9]
    tag_cols    = [16, 17, 18]
    contact_col = -2 # Second to last column is contact boolean
    trigger_col = -1 # Last column is trigger boolean

    for i, row in enumerate(csv_data):
        # Time is in first column
        time[i] = float(row[0])
        # Extract Force data
        f_exp_raw[i, :]   = [float(row[j]) for j in f_cols]
        # Extract EE XYZ position
        ee_exp[i, :]  = [float(row[j]) for j in ee_cols]
        # Tag roll, pitch, yaw
        tag_exp_raw[i, :] = [float(row[j]) for j in tag_cols]
        # Contact boolean
        cnt_exp[i] = int(row[contact_col])
        # Trigger boolean
        trig_exp[i] = int(row[trigger_col])

    time -= time[0]  # Normalize time to start at zero

    # Bias tag data to start at zero by subtracting the mean of the last 10 samples per column
    tag_exp_raw = tag_exp_raw - np.mean(tag_exp_raw[-10:, :], axis=0)
    # Throw away all values for tag angles that are above 60 degrees (clearly bad data)
    for i in range(3):
        bad_idx = np.where(np.abs(tag_exp_raw[:, i]) > np.deg2rad(60))[0]
        tag_exp_raw[bad_idx, i] = 0.0

    # END EFFECTOR is offset +3.5cm in z-axis due to mounting of robot
    # ee_exp[:, 2] -= 0.035  # - 0.035 Adjust Z position of EE for offset

    # Apply static transform to get Force XYZ correctly oriented
    f_temp = f_exp_raw.copy()
    f_exp_raw[:, 0] = f_temp[:, 2]  # X_force = Z_sensor
    f_exp_raw[:, 1] = f_temp[:, 0]  # Y_force = X_sensor
    f_exp_raw[:, 2] = f_temp[:, 1]  # Z_force = Y_sensor

    contact_idx_og  = np.where(cnt_exp == 1)[0][0]  # first index where contact boolean is true
    settle_idx_og = np.where(trig_exp == 1)[0][0]
    if shape == "box":
        settle_idx_og = contact_idx_og + 2000  # TEMP HACK: assume settles 4 seconds after contact (500 Hz)

    ## ================ Process the data (specifically force) ===================
    # Butterworth filter
    b, a        = butter(4, 5, fs=500, btype='low') # 4,5,500 : order, cutoff freq (<0.5*fs), sampling freq
    # f_exp_filt  = filtfilt(b, a, f_exp_raw, axis=0)
    f_exp_med   = medfilt(f_exp_raw, kernel_size=(5,1))
    k_sg        = int(0.09*500)|1
    k_sg        = 89
    force_exp   = savgol_filter(f_exp_med, k_sg, polyorder=3, axis=0)

    # TODO: I also have to estimate/calculate e_hat from experiment instead of manually entering it

    # Now we can also filter the tag data
    tag_exp_filt = filtfilt(b, a, tag_exp_raw, axis=0)
    # Find the tip axis by seeing which column has the max slope at ANY time (sometimes angle wrapping makes wrong column start huge)
    # Let's only look at the analysis window between contact and settle
    tag_exp_window = tag_exp_filt[contact_idx_og:settle_idx_og, :]
    scores = np.zeros(3)
    for k in range(3):
        theta = tag_exp_window[:, k]

        # Mask out wrap spikes / unphysical jumps
        good = np.abs(theta) < np.deg2rad(60)  # only consider angles less than 60 degrees
        if np.sum(good) < 10:
            scores[k] = 0
            continue
        theta_good = theta[good]
        scores[k] = np.max(theta_good) - np.min(theta_good)

    TIP_AXIS = np.argmax(scores)
    # TIP_AXIS = np.argmax(np.abs(tag_exp_filt[-1, :] - tag_exp_filt[0, :])) # compare start and end values to get primary axis
    print(f"\nPrimary tipping axis is {['roll', 'pitch', 'yaw'][TIP_AXIS]} (col {TIP_AXIS})")
    
    theta_exp = tag_exp_filt[:, TIP_AXIS]

    # ================ Find contact, settling, and start Indexes ===================
    # TEMP HACK: add buffer before (not after!) contact to capture correct initial force as it drives mass estimate
    # contact_idx_og -= 50

    # Get corresponding times
    contact_time_og = time[contact_idx_og]
    settle_time_og  = time[settle_idx_og]

    time_trim   = np.array(time[contact_idx_og:settle_idx_og])
    f_trim      = np.array(force_exp[contact_idx_og:settle_idx_og, :])
    th_trim     = np.array(theta_exp[contact_idx_og:settle_idx_og])
    ee_trim     = np.array(ee_exp[contact_idx_og:settle_idx_og, :])

    print(f"Contact at index {contact_idx_og} ({contact_time_og:.2f} s)")
    print(f"Settles at index {settle_idx_og} ({settle_time_og:.2f} s)")

## ================ Plot raw data ===================
    if plot_raw:
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax2 = plt.twinx()
        ax.plot(time, f_exp_raw[:, 0], "b", linewidth=2, label='X Force (raw)')
        ax.plot(time, f_exp_raw[:, 1], "r", linewidth=2, label='Y Force (raw)')
        ax.plot(time, f_exp_raw[:, 2], "m", linewidth=2, label='Z Force (raw)')
        ax.axvline(contact_time_og, color='k', linestyle='--', linewidth=2, label='Contact & Settle')
        ax.axvline(settle_time_og, color='k', linestyle='--', linewidth=2, label='_')
        ax2.plot(time, np.rad2deg(tag_exp_raw[:, 0]), color='g', linestyle='-', linewidth=3, label='Object angle (raw)')
        ax2.plot(time, np.rad2deg(tag_exp_raw[:, 1]), color='g', linestyle='-', linewidth=3, label='Object angle (raw)')
        ax2.plot(time, np.rad2deg(tag_exp_raw[:, 2]), color='g', linestyle='-', linewidth=3, label='Object angle (raw)')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax2.tick_params(axis='y', labelcolor='g', labelsize=20)
        ax.set_xlabel('Time (s)', fontsize=15)
        ax.set_ylabel('Force (N)', color='b', fontsize=15)
        ax2.set_ylabel('Angle (deg)', color='g', fontsize=15)
        ax.grid(True)

        # FOR PAPER FIGURE, TRIM THE AXES LIMITS
        # ax.set_xlim(1.6, 6)
        # ax2.set_ylim(0, 30)

        PLOT_FILT = True
        if PLOT_FILT:
            ax.plot(time, force_exp[:, 0], color='k', linestyle='-', linewidth=2, label='Filtered Force')
            ax.plot(time, force_exp[:, 1], color='k', linestyle='-', linewidth=2, label='_')
            ax.plot(time, force_exp[:, 2], color='k', linestyle='-', linewidth=2, label='_')

        fig.legend(loc='upper left', fontsize=15)
        align_zeros([ax, ax2])
        plt.tight_layout()


    if plot_ee:
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax2 = plt.twinx()
        ax.plot(time, ee_exp[:, 0], "b", linewidth=3, label='X Pos. (raw)')
        ax.plot(time, ee_exp[:, 1], "r", linewidth=3, label='Y Pos. (raw)')
        ax.plot(time, ee_exp[:, 2], "m", linewidth=3, label='Z Pos. (raw)')
        ax.axvline(contact_time_og, color='k', linestyle='--', linewidth=2, label='_')
        ax.axvline(settle_time_og, color='k', linestyle='--', linewidth=2, label='_')
        # ax2.plot(time, np.rad2deg(tag_exp_raw[:, 0]), color='g', linestyle='-', linewidth=3, label='Object angle (raw)')
        # ax2.plot(time, np.rad2deg(tag_exp_raw[:, 1]), color='g', linestyle='-', linewidth=3, label='Object angle (raw)')
        ax2.plot(time, np.rad2deg(tag_exp_raw[:, TIP_AXIS]), color='g', linestyle='-', linewidth=3, label='Object angle (raw)')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax2.tick_params(axis='y', labelcolor='g', labelsize=20)
        ax.set_xlabel('Time (s)', fontsize=15)
        ax.set_ylabel('Position (m)', color='b', fontsize=15)
        ax2.set_ylabel('Angle (rad)', color='g', fontsize=15)
        fig.legend(loc='upper left', fontsize=15)
        ax.grid(True)

        # FOR PAPER FIGURE, TRIM THE AXES LIMITS
        # ax.set_xlim(1.6, 6)
        # ax2.set_ylim(0, 30)

        align_zeros([ax, ax2])
        plt.tight_layout()


    # =============== Analysis on trimmed data ===================
    # Find index of max force (after init transients)
    f_trim_norm     = np.linalg.norm(f_trim.copy(), axis=1)
    
    peaks, _        = find_peaks(f_trim_norm, height=0, prominence=0.1, distance=20)
    if len(peaks) == 0:
        print(f"No peaks found in force magnitude data! Falling back to max value.")
        fmax_idx    = np.argmax(f_trim_norm, axis=0)  # Fallback to max if no peaks found
        fmax        = np.max(f_trim_norm)
    else:
        print(f"Found {len(peaks)} peaks in force magnitude data at indices: {peaks}")
        fmax_idx    = peaks[-1]  # Use last peak as fmax_idx

    print(f"F_max time: {time_trim[fmax_idx]:.2f} s, corresponding angle: {np.rad2deg(th_trim[fmax_idx]):.2f} degrees\n")

    ## =================== HACK: Modify experiment parameters to make better fitting ===================
    # # Don't currently have exact measure for o_obj. HOWEVER, at theta_crit, the EE x-pos should be equal to o x-pos!
    # ee_at_theta_star = ee_trim[np.argmin(np.abs(th_trim - theta_star_calc)), :] + 0.04 # Small 4cm offset makes it match!! TODO: Investigate tiny error propagation
    # o_obj = np.array([ee_at_theta_star[0], 0, 0])
    tool_flange = 0.47065
    ft_finger   = 0.081 + 0.114
    o_obj = np.array([tool_flange + ft_finger, 0, 0]) # Tool flange pos (NEW from FlexPendent + FT + Finger)
    # o_obj = np.array([0.66, 0, 0]) # NEW GROUND TRUTH BASED ON MEASUREMENT (this is really close to what we calced above)
    print(f"\nUsing o_obj = {o_obj} for analysis")

    # Amplify force a bit since we lose some due to friction in the FT sensor and robot
    # FRICTION_LOSS_FACTOR = 1.5 # 1.15
    # f_trim *= FRICTION_LOSS_FACTOR
    ## =================================================================================================

    
    ## ================== CALCULATE THETA* AND ZC USING LINEAR FIT ==================
    # Fit straight line to force data to find zero crossing
    lin_slope, lin_b, _, _, _ = linregress(th_trim[fmax_idx:], f_trim[fmax_idx:, 0])
    theta_star_calc = -lin_b / lin_slope
    zc_calc = abs(com_gt[0]) / np.tan(theta_star_calc)
    m_calc = abs(lin_slope)*ee_trim[-1,2]/(9.81*zc_calc)

    print(f"\nRecall init guess from linear fit:")
    print(f"mass: {m_calc:.3f} kg (GT: {m_gt:.3f} kg)")
    print(f"zc: {zc_calc:.3f} m (GT: {com_gt[2]:.3f} m)")
    print(f"theta*: {np.rad2deg(theta_star_calc):.2f} deg (GT: {theta_star_gt:.2f} deg)")
    
    ## =============== Fit using TAU model ==================
    rc0_known       = np.array([com_gt[0], com_gt[1], 0.0])
    rf              = (ee_trim - o_obj)

    # TEMP: Try replacing final data point with expected behavior (0 torque at theta* from linear fit)
    # f_trim[-1,:] = 0.0
    # th_trim[-1] = theta_star_calc
    
    # Before fitting, must pre-compute corresponding PUSH torque
    f_app = -f_trim # NOTE: IMPORTANT NEGATE TO MATCH F OBJECT EXPERIENCES

    # TEMP HACK: FRICTION IN Z COMPONENT IS PARASITIC FOR TORQUE, SO ZERO IT OUT FOR NOW...
    f_app[:, 1] = 0.0
    f_app[:, 2] = 0.0
    tau_app_trim    = tau_app_model(f_app, rf)

    # Try augmenting with a pseudo point at (theta*, tau=0) to help guide fit
    # th_trim         = np.concatenate([th_trim, [theta_star_calc]])
    # tau_app_trim    = np.concatenate([tau_app_trim,   [0.0]])
    # # weights: small sigma => high weight
    # w_theta         = 0.000001   # tune this: 0.1 = very soft, 1.0 = as strong as a real point
    # sigma           = np.ones_like(tau_app_trim)  # real points: σ = 1
    # sigma[-1]       = 1.0 / w_theta          # pseudo point: weight ≈ w_theta^2

    # time_trim = np.concatenate([time_trim, [time_trim[-1]+1.5]]) # small time extension for pseudo point
    # f_trim = np.concatenate([f_trim, [[0.0, 0.0, 0.0]]]) # pseudo zero force point
    # rf = np.concatenate([rf, [[0.0, 0.0, 0.0]]]) # pseudo zero lever arm point

    [m_est, zc_est], pcov = curve_fit(
        lambda th, m, zc: tau_model(th, m, zc, rc0_known=rc0_known),
        th_trim,
        tau_app_trim,
        p0=[m_calc, zc_calc],
        # sigma=sigma,
        # absolute_sigma=False,
        bounds=([0, 0],
                [np.inf, np.inf])
        )

    # Now use fitted parameters to estimate theta_star
    theta_star_est = np.arctan2(np.linalg.norm(com_gt[:1]), zc_est) # atan2( d (xy norm), z) 

    ## ================== PLOTTING THE TORQUE DIRECTLY ===================
    th_extrap_est = np.linspace(0, theta_star_est, len(ee_trim))

    fig, ax = plt.subplots(figsize=(8,4.5))
    # Plot the original data
    tau_app_trim = tau_app_trim.reshape(-1,3)
    # ax.plot(np.rad2deg(th_trim), tau_app_trim[:,0], 'gray', linewidth=5, label='Original tau x')  # Plot tau
    ax.plot(np.rad2deg(th_trim), tau_app_trim[:,1], 'k', linewidth=5, label='Original tau y')  # Plot tau
    # ax.plot(np.rad2deg(th_trim), tau_app_trim[:,2], 'gray', linewidth=5, label='Original tau z')  # Plot tau
    # Plot our model using fitted params
    tau_model_est = tau_model(th_trim, m_est, zc_est, rc0_known=rc0_known).reshape(-1,3)
    # ax.plot(np.rad2deg(th_trim), tau_model_est[:,0], color='b', linestyle='-', linewidth=3, label='Model fit (experienced) X')
    ax.plot(np.rad2deg(th_trim), tau_model_est[:,1], color='b', linestyle='-', linewidth=3, label='Model fit (experienced) Y')
    # ax.plot(np.rad2deg(th_trim), tau_model_est[:,2], color='b', linestyle='-', linewidth=3, label='Model fit (experienced) Z')
    # Plot our model but to extrapolated full theta range
    tau_model_full_est = tau_model(th_extrap_est, m_est, zc_est, rc0_known=rc0_known).reshape(-1,3)
    # ax.scatter(np.rad2deg(th_extrap_est), tau_model_full_est[:,0], color='m', label='Full fit (est) X')
    ax.scatter(np.rad2deg(th_extrap_est), tau_model_full_est[:,1], color='m', label='Full fit (est) Y')
    # ax.scatter(np.rad2deg(th_extrap_est), tau_model_full_est[:,2], color='m', label='Full fit (est) Z')
    # Plot using the ground truth params for reference
    tau_model_full_gt = tau_model(th_extrap_est, m_gt, com_gt[2], rc0_known=rc0_known).reshape(-1,3)
    ax.scatter(np.rad2deg(th_extrap_est), tau_model_full_gt[:,1], color='c', label='Full fit (gt) X')
    # Let's also plot the 'rf' to see the lever arm change as the object rotates/tips
    # ax2 = plt.twinx()
    # ax2.plot(np.rad2deg(th_trim), np.linalg.norm(rf, axis=1), color='orange', linestyle='-', linewidth=2, label='Lever arm norm')

    # Scatter the Estimated and Ground Truth theta*
    ax.axvline(theta_star_gt, color='g', linestyle='--', linewidth=5, label=r'Ground truth $\theta^*$')
    ax.scatter(np.rad2deg(theta_star_est), 0, s=500, marker='*', color='m', label=r'Estimated $\theta^*$', zorder=2)

    ax.axhline(0, color='c', linewidth=2) # Horizontal line at zero for reference
    ax.set_ylabel("Applied Torque (N-m)", color='b', fontsize=20)
    ax.set_xlabel("Object Angle (degrees)", color='g', fontsize=20)
    ax.legend(loc='upper right', fontsize=15)
    ax.grid(True)
    ax.tick_params(axis='y', labelcolor='b', labelsize=20)
    ax.tick_params(axis='x', labelcolor='g', labelsize=20)
    plt.tight_layout()


    ## ================== PLOTTING THE FIT RESULTS ===================
    # Plot our model using fitted params ONLY for the experienced data range
    f_app_model_est = F_model(th_trim, m_est, zc_est, rf, rc0_known=rc0_known)
    # Extrapolate experienced data to mimic a full toppling experiment for plotting
    f_app_full_est = F_model(th_extrap_est, m_est, zc_est, rf, rc0_known=rc0_known)
    f_app_full_gt  = F_model(th_extrap_est, m_gt, com_gt[2], rf, rc0_known=rc0_known)

    # Whether to use the primary tipping and push axes (for plotting too!)
    PLOT_X_ONLY = True # False # NORM LOOKS BAD!!!

    ## IMPORTANT: must negate the 'experienced force' back again!
    f_app_model_est *= -1
    f_app_full_est  *= -1
    f_app_full_gt   *= -1

    if PLOT_X_ONLY:
        f_plot_exp = f_trim[:,0]
        y_label = "X-Force (N)"
        f_plot_model_est = f_app_model_est[:,0]
        f_plot_full_est = f_app_full_est[:,0]
        f_plot_full_gt = f_app_full_gt[:,0]
    else:
        f_plot_exp = -np.linalg.norm(f_trim, axis=1)
        lin_slope, lin_b, _, _, _ = linregress(th_trim[fmax_idx:], -np.linalg.norm(f_trim[fmax_idx:], axis=1))
        theta_star_calc = -lin_b / lin_slope
        y_label = "Force Norm (N)"
        f_plot_model_est = -np.linalg.norm(f_app_model_est, axis=1)
        f_plot_full_est = -np.linalg.norm(f_app_full_est, axis=1)
        f_plot_full_gt = -np.linalg.norm(f_app_full_gt, axis=1)

    # Plot the linear fit using y = mx + b
    th_extrap_calc = np.linspace(0, theta_star_calc, len(ee_trim))
    f_plot_lin_calc = lin_slope * th_extrap_calc + lin_b


    ## ============================ PLOTTING ============================
    fig, ax = plt.subplots(figsize=(8,4.5))
    # Plot the original data
    ax.plot(np.rad2deg(th_trim), f_plot_exp, 'k', linewidth=5, label='Original data')  # Plot F norm

    # Plot linear fit using theta_star_calc from prior linear fit
    ax.plot(np.rad2deg(th_extrap_calc), f_plot_lin_calc, color='r', linestyle='--', linewidth=3, label='Linear fit')

    # Plot our model using fitted params for the experienced data range
    ax.plot(np.rad2deg(th_trim), f_plot_model_est, color='b', linestyle='-', linewidth=3, label='Model fit (experienced)')

    ## FOR FUN, plot ALL theta and force
    ax.scatter(np.rad2deg(th_extrap_est), f_plot_full_est, color='m', label='Full fit (est)')
    ax.scatter(np.rad2deg(th_extrap_est), f_plot_full_gt, color='c', label='Full fit (gt)')

    # Scatter the Estimated, Linear (calc), and Ground Truth theta*
    ax.axvline(theta_star_gt, color='g', linestyle='--', linewidth=5, label=r'Ground truth $\theta^*$')
    ax.scatter(np.rad2deg(theta_star_calc), 0, s=500, marker='*', color='r', label=r'Linear Guess $\theta^*$', zorder=2)
    ax.scatter(np.rad2deg(theta_star_est), 0, s=500, marker='*', color='m', label=r'Estimated $\theta^*$', zorder=2)

    # TEMP HACK: Plot all 3 force axes for reference
    ax.plot(np.rad2deg(th_trim), f_app_full_est[:,1], 'r', linewidth=1, label='Y Force (trimmed)')
    ax.plot(np.rad2deg(th_trim), f_app_full_est[:,2], 'm', linewidth=1, label='Z Force (trimmed)')

    ax.axhline(0, color='c', linewidth=2) # Horizontal line at zero for reference
    ax.set_ylabel(y_label, color='b', fontsize=20)
    ax.set_xlabel("Object Angle (degrees)", color='g', fontsize=20)
    ax.legend(loc='lower right', fontsize=15)
    ax.grid(True)
    ax.tick_params(axis='y', labelcolor='b', labelsize=20)
    ax.tick_params(axis='x', labelcolor='g', labelsize=20)
    plt.tight_layout()

    # ================ Final printout of results ==================
    print(f"\nFitted results:")
    print(f"mass: {m_est:.3f} kg (GT: {m_gt:.3f} kg)")
    print(f"zc: {zc_est:.3f} m (GT: {com_gt[2]:.3f} m)")
    print(f"theta*: {np.rad2deg(theta_star_est):.2f} deg (GT: {theta_star_gt:.2f} deg)")

    print(f"\n^^^^^^^^ END Analysis on Shape: {shape.upper()}^^^^^^^^\n")

    # plt.show(block=False)
    # input("Press Enter to continue...")

    return m_est, zc_est, theta_star_est


if __name__ == "__main__":
    # Close all previously open plots
    plt.close('all')

    # ================ Load the data ===================
    # path = "experiments/run_2025-11-10_15-28-42_t001_SYNC.csv"
    # path = "experiments/run_2025-11-10_15-52-45_t002_SYNC.csv" # Best
    # path = "experiments/run_2025-11-10_20-00-09_t002_SYNC.csv" # Best

    shapes_to_run = ["box", "heart", "flashlight"] #, "lshape"]#, "monitor"]

    for shape in shapes_to_run:
        obj     = OBJECTS[shape]
        path    = obj["path"]
        com     = obj["com"]
        m_gt       = obj["mass"]
        # Ground truth from geometry
        theta_star_gt = np.rad2deg(np.arctan2(abs(np.linalg.norm(com[0:1])), com[2]))

        print(f"\n========= Analyzing object: {shape.upper()} =========")
        print(f"Ground truth from geometry:\ntheta*: {theta_star_gt:.2f} deg, zc = {com[2]:.3f} m")

        m_est, zc_est, theta_star_est = main(shape, path, com, m_gt, theta_star_gt, plot_raw=True, plot_ee=False)

        obj["est"] = [m_est, zc_est, theta_star_est]


    print("\n\n========= FINAL ESTIMATES ========= ")
    for shape in shapes_to_run:
        obj = OBJECTS[shape]
        com = obj["com"]
        m_gt   = obj["mass"]
        height = obj["height"]
        m_est, zc_est, theta_star_est = obj["est"]
        print(f"{shape.upper()}:")
        print(f"Estimated: mass = {m_est:.3f} kg, zc = {zc_est:.3f} m, theta* = {np.rad2deg(theta_star_est):.2f} deg")
        print(f"Gnd Truth: mass = {m_gt:.3f} kg, zc = {com[2]:.3f} m, theta* = {theta_star_gt:.2f} deg")
        # Error as a percentage of mass and object height
        m_err = m_est - m_gt
        m_err_pct = abs(m_err)/m_gt * 100
        zc_err = zc_est - com[2]
        zc_err_pct = abs(zc_err)/height * 100
        print(f"   Errors: mass = {m_err:.3f} kg, {m_err_pct:.2f} %, zc = {zc_err:.3f} m, {zc_err_pct:.2f} %")

    plt.show()