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



def read_csv(file_path, trim_rows=0):
    with open(file_path, mode ='r')as file:
        csvFile = csv.reader(file)
        csv_arr = []
        for lines in csvFile:
                csv_arr.append(lines)
        return csv_arr[trim_rows:]


def main(csv_path, com_gt, m_gt, topple=False):

    csv_data = read_csv(csv_path, trim_rows=1)  # Discard headers


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

    # Bias tag data to start at zero by subtracting the mean of the first 10 samples per column
    tag_exp_raw = tag_exp_raw - np.mean(tag_exp_raw[:10, :], axis=0)

    # END EFFECTOR is offset +3.5cm in z-axis due to mounting of robot
    ee_exp[:, 2] -= 0.035  # Adjust Z position of EE for offset

    # Apply static transform to get Force XYZ correctly oriented
    temp = f_exp_raw.copy()
    f_exp_raw[:, 0] = temp[:, 2]  # X_force = Z_sensor
    f_exp_raw[:, 1] = temp[:, 0]  # Y_force = X_sensor
    f_exp_raw[:, 2] = temp[:, 1]  # Z_force = Y_sensor



    ## ================ Process the data (specifically force) ===================
    # Butterworth filter
    b, a        = butter(4, 5, fs=500, btype='low') # 4,5,500 : order, cutoff freq (<0.5*fs), sampling freq
    # f_exp_filt  = filtfilt(b, a, f_exp_raw, axis=0)
    f_exp_med   = medfilt(f_exp_raw, kernel_size=(5,1))
    k_sg        = int(0.09*500)|1
    k_sg        = 89
    f_exp_filt  = savgol_filter(f_exp_med, k_sg, polyorder=3, axis=0)

    # TODO: I also have to estimate/calculate e_hat from experiment instead of manually entering it

    # Now we can also filter the tag data
    tag_exp_filt = filtfilt(b, a, tag_exp_raw, axis=0)
    TIP_AXIS = np.argmax(np.abs(tag_exp_filt[-1, :] - tag_exp_filt[0, :])) # compare start and end values to get primary axis
    print(f"\nPrimary tipping axis is {['roll', 'pitch', 'yaw'][TIP_AXIS]} (col {TIP_AXIS})")
    
    th_exp = tag_exp_filt[:, TIP_AXIS]

    # ================ Find contact, settling, and start Indexes ===================
    contact_idx_og  = np.where(cnt_exp == 1)[0][0]  # first index where contact boolean is true
    settle_idx_og = np.where(trig_exp == 1)[0][0]
    
    # Get corresponding times
    contact_time_og = time[contact_idx_og]
    settle_time_og  = time[settle_idx_og]

    # =============== Trim the data to the analysis window ===================
    # Add a buffer around contact and settling for analysis window
    # ANALYSIS_WINDOW_BUFFER = 0.1 # seconds
    # start_idx_og = np.where(time >= contact_time_og - ANALYSIS_WINDOW_BUFFER)[0][0]
    # end_idx_og = np.where(time >= settle_time_og + ANALYSIS_WINDOW_BUFFER)[0][0]

    # time_trim = np.array(time[start_idx_og:end_idx_og])
    # f_trim = np.array(f_exp_filt[start_idx_og:end_idx_og, :])
    # th_trim = np.array(th_exp[start_idx_og:end_idx_og])
    # ee_trim = np.array(ee_exp[start_idx_og:end_idx_og, :])

    time_trim = np.array(time[contact_idx_og:settle_idx_og])
    f_trim = np.array(f_exp_filt[contact_idx_og:settle_idx_og, :])
    th_trim = np.array(th_exp[contact_idx_og:settle_idx_og])
    ee_trim = np.array(ee_exp[contact_idx_og:settle_idx_og, :])

    print(f"Contact at index {contact_idx_og} ({contact_time_og:.2f} s)")
    print(f"Settles at index {settle_idx_og} ({settle_time_og:.2f} s)")
    # print(f"With bufffer: {time_trim[0]:.2f} s to {time_trim[-1]:.2f} s\n")

## ================ Plot raw data ===================
    PLOT_RAW = True

    if PLOT_RAW:
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax2 = plt.twinx()
        ax.plot(time, f_exp_raw[:, 0], "b", linewidth=3, label='X Force (raw)')
        ax.plot(time, f_exp_raw[:, 1], "r", linewidth=3, label='Y Force (raw)')
        ax.plot(time, f_exp_raw[:, 2], "m", linewidth=3, label='Z Force (raw)')
        ax.axvline(contact_time_og, color='k', linestyle='--', linewidth=2, label='Contact & Settle')
        ax.axvline(settle_time_og, color='k', linestyle='--', linewidth=2, label='_')
        # ax2.plot(time, np.rad2deg(tag_exp_raw[:, 0]), color='g', linestyle='-', linewidth=3, label='Object angle (raw)')
        # ax2.plot(time, np.rad2deg(tag_exp_raw[:, 1]), color='g', linestyle='-', linewidth=3, label='Object angle (raw)')
        ax2.plot(time, np.rad2deg(tag_exp_raw[:, TIP_AXIS]), color='g', linestyle='-', linewidth=3, label='Object angle (raw)')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax2.tick_params(axis='y', labelcolor='g', labelsize=20)
        ax.set_xlabel('Time (s)', fontsize=15)
        ax.set_ylabel('Force (N)', color='b', fontsize=15)
        ax2.set_ylabel('Angle (rad)', color='g', fontsize=15)
        ax.grid(True)

        # FOR PAPER FIGURE, TRIM THE AXES LIMITS
        # ax.set_xlim(1.6, 6)
        # ax2.set_ylim(0, 30)

        PLOT_FILT = True
        if PLOT_FILT:
            ax.plot(time, f_exp_filt[:, 0], color='k', linestyle='-', linewidth=2, label='Filtered Force')
            ax.plot(time, f_exp_filt[:, 1], color='k', linestyle='-', linewidth=2, label='_')
            ax.plot(time, f_exp_filt[:, 2], color='k', linestyle='-', linewidth=2, label='_')

        fig.legend(loc='upper left', fontsize=15)
        align_zeros([ax, ax2])
        plt.tight_layout()

    PLOT_EE = False

    if PLOT_EE:
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
    f_trim_norm = np.linalg.norm(f_trim.copy(), axis=1)
    
    peaks, _ = find_peaks(f_trim_norm, height=0, prominence=0.1, distance=20)
    if len(peaks) == 0:
        print(f"No peaks found in force magnitude data! Falling back to max value.")
        fmax_idx = np.argmax(f_trim_norm, axis=0)  # Fallback to max if no peaks found
    else:
        print(f"Found {len(peaks)} peaks in force magnitude data at indices: {peaks}")
        fmax_idx = peaks[-1]  # Use last peak as fmax_idx

    print(f"F_max time: {time_trim[fmax_idx]:.2f} s, corresponding angle: {np.rad2deg(th_trim[fmax_idx]):.2f} degrees\n")


    # Ground truth from geometry
    theta_star_gt = np.rad2deg(np.arctan2(abs(np.linalg.norm(com_gt[:1])), com_gt[2]))
    print(f"Ground truth from geometry:\ntheta*: {theta_star_gt:.2f} deg, zc = {com_gt[2]:.3f} m")
    
    # Calculate theta* and zc from experiment
    print("******* SUB-CRIT PUSH: ESTIMATE ZERO-CROSSING w/ LINEAR FIT **************")
    # Fit straight line to force data to find zero crossing
    lin_slope, lin_b, _, _, _ = linregress(th_trim[fmax_idx:], f_trim[fmax_idx:, 0])
    ## ==================== CALCULATE THETA* AND ZC ====================
    theta_star_calc = -lin_b / lin_slope
    f0_idx = np.argmin(np.abs(th_trim - theta_star_calc))
    zc_calc = abs(com_gt[0]) / np.tan(theta_star_calc)
    print(f"Calculated from linear fit:\ntheta* = {np.rad2deg(theta_star_calc):.2f} deg, zc = {zc_calc:.3f} m")


    # ================ Plot the data ===================
    PLOT_RELATIONSHIP = False        

    if PLOT_RELATIONSHIP:
        # NOTE: This is plotting the primary pushing force, NOT the magnitude.
        # Now plot force (in primary tipping axis) versus payload pitch
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax.axhline(0, color='c') # Horizontal line at zero for reference
        ax.plot(np.rad2deg(th_trim[:f0_idx+100]), f_trim[:f0_idx+100, 0], \
                color='k', linewidth=5, label='Push force (x)')  # Plot the x-component of the force (up to 100 indices after zero-crossing)
        ax.set_ylabel("X-Force (N)", color='b', fontsize=20)
        ax.set_xlabel("Object Angle (deg)", color='g', fontsize=20)
        ax.axvline(theta_star_gt, color='g', linestyle='--', linewidth=5, label=r'Ground truth $\theta^*$')
        ax.scatter(np.rad2deg(theta_star_calc), 0, s=500, marker='*', color='r', label=r'Calculated $\theta^*$', zorder=2)
        # Plot Linear fit
        lin_plot_deg = np.linspace(0, theta_star_calc, 100)
        ax.plot(np.rad2deg(lin_plot_deg), lin_slope * lin_plot_deg + lin_b, color='r', linestyle='--', linewidth=3, label='Linear fit')
        
        ax.grid(True)
        ax.legend(loc='upper left', fontsize=15) # # Now also plot a * at zero-crossing

        ax.tick_params(axis='y', labelcolor='b', labelsize=20)
        ax.tick_params(axis='x', labelcolor='g', labelsize=20)
        plt.tight_layout()

        PLOT_INSERT = False

        if PLOT_INSERT:
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


    ## =================== HACK: Modify experiment parameters to make better fitting ===================
    print("\n ************** SUB CRITICAL FIT CALCULATION ****************")
    # # Don't currently have exact measure for o_obj. HOWEVER, at theta_crit, the EE x-pos should be equal to o x-pos!
    ee_at_theta_star = ee_trim[np.argmin(np.abs(th_trim - theta_star_calc)), :] + 0.04 # Small 4cm offset makes it match!! TODO: Investigate tiny error propagation
    o_obj = np.array([ee_at_theta_star[0], 0, 0])

    # o_obj = np.array([0.398, 0, 0]) # Experimentally determined for now

    # o_obj = 0.47065 # Tool flange pos from tablet
    # +81+114mm for FT and finger
    print(f"\nUsing o_obj = {o_obj} for analysis")


    # ALSO, my mass is being estimated too low... Let's try artificially boosting the force to see if theres some factor there
    # f_subcrit *= 1.12

    ## =================================================================================================

    f_app_subcrit = -f_trim # NOTE: IMPORTANT NEGATE TO MATCH F OBJECT EXPERIENCES
    
    m_guess = abs(lin_slope)*ee_trim[-1,2]/(9.81*zc_calc)
    print(f"\nRecall init guess from linear fit:")
    print(f"mass: {m_guess:.3f} kg    zc: {zc_calc:.3f} m    theta*: {np.rad2deg(theta_star_calc):.3f} deg")
    

    ## =============== Fit using TAU model ==================
    # Before fitting, must pre-compute corresponding PUSH torque
    tau_app_subcrit = tau_app_model(f_app_subcrit, (ee_trim - o_obj)).ravel()

    [m_est, zc_est], pcov = curve_fit(tau_model, th_trim, tau_app_subcrit, p0=[m_guess, zc_calc])
    # Now use fitted parameters to estimate theta_star
    theta_star_est = np.arctan2(np.linalg.norm(com_gt[:1]), zc_est) # atan2( d (xy norm), z)

    # Extrapolate experienced data to mimic a full toppling experiment for plotting
    th_full = np.linspace(th_trim[0], theta_star_calc, len(ee_trim))
    f_app_full = F_model(th_full, m_est, zc_est, (ee_trim - o_obj))
    # Plot the linear fit using y = mx + b
    f_app_lin = lin_slope * th_full + lin_b
    

    # Whether to use the primary tipping and push axes (for plotting too!)
    PLOT_X_ONLY = True
    # PLOT_X_ONLY = False

    if PLOT_X_ONLY:
        f_plot_exp = f_trim[:,0]
        f_plot_sc = -f_app_subcrit[:,0]
        f_plot_lin = f_app_lin
        f_plot_full = -f_app_full[:,0]
        y_label = "X-Force (N)"
    else:
        f_plot_exp = np.linalg.norm(f_trim, axis=1)
        f_plot_sc = -np.linalg.norm(f_app_subcrit, axis=1)
        f_plot_lin = -f_app_lin
        f_plot_full = np.linalg.norm(f_app_full, axis=1)
        y_label = "Force Norm (N)"


    ## ============================ PLOTTING ============================
    ## Plot original data and sub-critical window
    fig, ax = plt.subplots(figsize=(8,4.5))
    # Plot the original data
    ax.plot(np.rad2deg(th_trim), f_plot_exp, 'k', linewidth=5, label='Original data')  # Plot F norm
    # Plot the sub-critical data
    # ax.scatter(np.rad2deg(th_subcrit), f_plot_sc, color='r', s=200, alpha=0.9, label='Sub-critical window')
    # Plot the linear fit (already done in another plot above...)
    # ax.plot(np.rad2deg(th_full), f_plot_lin, color='orange', linewidth=3, label='Sub-crit linear fit')
    ## FOR FUN, plot ALL theta and force
    ax.scatter(np.rad2deg(th_full), f_plot_full, color='g', label='Full fit')
    ax.axhline(0, color='c', linewidth=2) # Horizontal line at zero for reference
    ax.axvline(theta_star_gt, color='g', linestyle='--', linewidth=5, label=r'Ground truth $\theta^*$')
    ax.scatter(np.rad2deg(theta_star_est), 0, s=500, marker='*', color='m', label=r'Estimated $\theta^*$', zorder=2)
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

    print(f"\nFrom estimated theta*, use geometry to back-check:")
    print(f"zc: {abs(com_gt[0])/np.tan(theta_star_est):.3f} m (GT: {com_gt[2]:.3f} m)")
        

    plt.show()


if __name__ == "__main__":
    
    # ================ Load the data ===================
    # path = "experiments/run_2025-11-10_15-28-42_t001_SYNC.csv"
    # path = "experiments/run_2025-11-10_15-52-45_t002_SYNC.csv" # Best
    # path = "experiments/run_2025-11-10_20-00-09_t002_SYNC.csv" # Best

    paths = []
    paths.append("experiments/20251208_155739_box_t02.csv") # Testing new force stop
    paths.append("experiments/20251208_155739_heart_t03.csv") # Testing new force stop
    # paths.append("experiments/20251208_143007_flashlight_t01.csv") # Just testing the weird force curve...
    paths.append("experiments/20251208_155739_flashlight_t04.csv") # Testing new force stop
    paths.append("experiments/20251208_155739_lshape_t03.csv") # Testing new force stop
    

    com_gt = [] # X and Y is distance from tipping edge (object frame!) to projected CoM on table plane
    com_gt.append([-0.0500, 0.0, 0.1500]) # box (onshape minus distance to e_hat tipping edge)
    com_gt.append([-0.0458, 0.0, 0.1000]) # heart
    com_gt.append([-0.0250, 0.0, 0.0950]) # flashlight
    com_gt.append([-0.0250, 0.0, 0.0887]) # lshape

    ## =================== SET THE GROUND TRUTH COM & MASS ===================
    m_gt = []
    m_gt.append(0.635)  # box
    m_gt.append(0.269)  # heart
    m_gt.append(0.386)  # flashlight
    m_gt.append(0.118)  # lshape
    

    item = 2

    main(paths[item], com_gt[item], m_gt[item], topple=False)