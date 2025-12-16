import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt, savgol_filter, find_peaks
from scipy.interpolate import interp1d
from scripts.utils.com_estimation import tau_app_model, tau_model, align_zeros, F_model
from scipy.stats import linregress
from scipy.spatial.transform import Rotation as R
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scripts.utils.helper_fns import quat_to_axis_angle, enforce_quat_continuity

OBJECTS = {
        # X and Y is distance from tipping edge (object frame!) to projected CoM on table plane
        "box": {
            # "path": "experiments/20251208_155739_box_t02.csv",
            "path": "experiments/20251215_180505_box_t01.csv",
            # "path": "experiments/20251215_180505_box_t02.csv",
            "com": [-0.0500, 0.0, 0.1500],
            "mass": 0.635,
            "theta_star": 0.0, # placeholder
            "height": 0.3, # 300 mm
            "est": [0,0,0],
        },
        "heart": {
            # "path": "experiments/20251208_155739_heart_t03.csv",
            "path": "experiments/20251215_182022_heart_t01.csv",
            # "path": "experiments/20251215_182022_heart_t02.csv",
            # "path": "experiments/20251215_182022_heart_t03.csv",
            "com": [-0.0458, 0, 0.0800], # [-0.0458, 0, 0.1]
            "mass": 0.269,
            "theta_star": 0.0, # placeholder
            "height": 0.2, # 200 mm
            "est": [0,0,0],
        },
        "flashlight": {
            # "path": "experiments/20251208_143007_flashlight_t01.csv",  # older test
            # "path": "experiments/20251208_155739_flashlight_t04.csv",    # new force stop test
            "path": "experiments/20251215_182909_flashlight_t01.csv",
            # "path": "experiments/20251215_182909_flashlight_t02.csv",
            # "path": "experiments/20251215_182909_flashlight_t03.csv",
            "com": [-0.0250, 0.0, 0.0950],
            "mass": 0.386,
            "theta_star": 0.0, # placeholder
            "height": 0.2, # 200 mm
            "est": [0,0,0],
        },
        "lshape": {
            # "path": "experiments/20251208_155739_lshape_t05.csv",
            "path": "experiments/20251215_182211_lshape_t01.csv",
            # "path": "experiments/20251215_182211_lshape_t02.csv",
            # "path": "experiments/20251215_182211_lshape_t03.csv",
            "com": [-0.0250, 0.0, 0.0887],
            "mass": 0.118,
            "theta_star": 0.0, # placeholder
            "height": 0.15, # 150 mm
            "est": [0,0,0],
        },
        "monitor": {
            # "path": "experiments/20251210_182429_monitor_t03.csv",
            "path": "experiments/20251215_183134_monitor_t01.csv",
            # "path": "experiments/20251215_183134_monitor_t02.csv",
            "com": [-0.1118, 0.0, 0.2362],
            "mass": 5.37,
            "theta_star": 0.0, # placeholder
            "height": 0.515, # 515 mm
            "est": [0,0,0],
        },
        "soda": {
            "path": "experiments/20251215_183134_soda_t01.csv",
            # "path": "experiments/20251215_183134_soda_t02.csv",
            "com": [-0.1118, 0.0, 0.2362],
            "mass": 5.37,
            "theta_star": 0.0, # placeholder
            "height": 0.515, # 515 mm
            "est": [0,0,0],
        },
    }

def read_csv(file_path, trim_rows=0):
    with open(file_path, mode ='r')as file:
        csvFile = csv.reader(file)
        csv_arr = []
        for lines in csvFile:
                csv_arr.append(lines)
        return csv_arr[trim_rows:]
    
def plot_3vec(ax, indep, vec, label, linewidth=3, linestyle='-', color_order=['b', 'r', 'm']):
    ax.plot(indep, vec[:, 0], color_order[0], linewidth=linewidth, linestyle=linestyle, label=f'X {label}')
    ax.plot(indep, vec[:, 1], color_order[1], linewidth=linewidth, linestyle=linestyle, label=f'Y {label}')
    ax.plot(indep, vec[:, 2], color_order[2], linewidth=linewidth, linestyle=linestyle, label=f'Z {label}')
    return ax

def set_fig_opts(ax, xlabel, ylabel, ax2=None, ylabel2=None):
    ax.tick_params(axis='x', labelsize=20, labelcolor='g')
    ax.tick_params(axis='y', labelsize=20, labelcolor='b')
    ax.set_xlabel(xlabel, fontsize=20, color='g')
    ax.set_ylabel(ylabel, fontsize=20, color='b')
    ax.grid(True)
    if ax2 is not None:
        ax2.tick_params(axis='y', labelsize=20, labelcolor='k')
        ax2.set_ylabel(ylabel2, fontsize=20, color='k')
        align_zeros([ax, ax2])
    plt.tight_layout()


def main(shape, csv_path, com_gt, m_gt, theta_star_gt, plot_raw=True, plot_ee=False):

    csv_data = read_csv(csv_path, trim_rows=1)  # Discard headers

    ## ================ Extract time series data ===================
    time        = np.zeros(len(csv_data))
    ee_exp      = np.zeros((len(csv_data), 3))
    f_exp_raw   = np.zeros((len(csv_data), 3)) # RAW because we are filtering later
    rvec_exp_raw = np.zeros((len(csv_data), 4)) # RAW because we are filtering later
    cnt_exp     = np.zeros(len(csv_data))      # Contact boolean
    trig_exp    = np.zeros(len(csv_data))      # Trigger boolean
    # Column nums for each data type: (time is column zero)
    f_cols      = [1, 2, 3]
    ee_cols     = [7, 8, 9]
    tag_cols    = [16, 17, 18, 19]
    contact_col = -2 # Second to last column is contact boolean
    trigger_col = -1 # Last column is trigger boolean

    for i, row in enumerate(csv_data):
        time[i] = float(row[0])
        f_exp_raw[i, :]   = [float(row[j]) for j in f_cols]
        ee_exp[i, :]  = [float(row[j]) for j in ee_cols]
        rvec_exp_raw[i, :] = [float(row[j]) for j in tag_cols]
        cnt_exp[i] = int(row[contact_col])
        trig_exp[i] = int(row[trigger_col])

    ## ================ Pre-process time, tag, and force data ===================
    time -= time[0]  # Normalize time to start at zero

    # Convert tag quaternions to rotation vector representation
    Q = enforce_quat_continuity(rvec_exp_raw)
    r_all = R.from_quat(Q)
    r_start = r_all[0:10].mean()
    # r_rel = r_start.inv() * r_all
    r_rel = r_all * r_start.inv()
    rot_vecs_cam = r_rel.as_rotvec() # rot vec magnitudes are angles in radians, unit direction is axis
    # Rotate camera to world frame
    R_cam_to_world = R.from_euler('x', -90, degrees=True)  # Rotate -90 deg about X axis
    rot_vecs_world = R_cam_to_world.apply(rot_vecs_cam)
    # Ignore out-of-plane (Z) rotations for tipping analysis
    rot_vecs_world[:, 2] = 0.0

    # Apply static transform to get Force XYZ correctly oriented
    f_temp = f_exp_raw.copy()
    f_exp_raw[:, 0] = f_temp[:, 2]  # X_force = Z_sensor
    f_exp_raw[:, 1] = f_temp[:, 0]  # Y_force = X_sensor
    f_exp_raw[:, 2] = f_temp[:, 1]  # Z_force = Y_sensor

    ## ================ Filter Force and Tag Data ===================
    # Butterworth filter
    b, a        = butter(4, 5, fs=500, btype='low') # 4,5,500 : order, cutoff freq (<0.5*fs), sampling freq
    f_exp_med   = medfilt(f_exp_raw, kernel_size=(5,1))
    k_sg        = 89
    force_exp   = savgol_filter(f_exp_med, k_sg, polyorder=3, axis=0)
    # Now we can also filter the tag data
    # rot_vecs_filt = filtfilt(b, a, rot_vecs_world, axis=0)
    rot_vecs_filt = savgol_filter(rot_vecs_world, k_sg, polyorder=3, axis=0)

    theta_exp = np.linalg.norm(rot_vecs_filt, axis=1)

    theta_safe = theta_exp.copy()
    theta_safe[theta_safe < 1e-6] = 1.0 # prevent div by zero
    axes_exp_raw = rot_vecs_filt / theta_safe[:, None] # "Tipping Axis" unit vectors

    noise_mask = theta_exp < np.deg2rad(0.5) # To prevent chatter in unit axis when angle is small
    axes_exp_raw[noise_mask, :] = 0.0
    theta_exp[noise_mask] = 0.0
    

    # ================ TRIM DATA TO ANALYSIS WINDOW ===================
    # Get contact and settle indexes from experiment
    contact_idx_og  = np.where(cnt_exp == 1)[0][0]  # first index where contact boolean is true
    settle_idx_og = np.where(trig_exp == 1)[0][0]
    # TEMP: Fix the incorrect contact and settle indexes for box shape... TODO: fix this in the experiment...
    if settle_idx_og <= contact_idx_og or shape == "box":
        contact_idx_og = 2350
        settle_idx_og = contact_idx_og + 1750
    
    if shape == "heart":
        contact_idx_og = int(4.1 * 500)  # 4.1 seconds
        settle_idx_og = int(6.4 * 500)  # 6.4 seconds
    
    # Get corresponding times
    contact_time_og = time[contact_idx_og]
    settle_time_og  = time[settle_idx_og]

    time_trim   = np.array(time[contact_idx_og:settle_idx_og])
    f_trim      = np.array(force_exp[contact_idx_og:settle_idx_og, :])
    th_trim     = np.array(theta_exp[contact_idx_og:settle_idx_og])
    ee_trim     = np.array(ee_exp[contact_idx_og:settle_idx_og, :])
    TIP_AXIS    = np.array(axes_exp_raw[contact_idx_og:settle_idx_og, :]).mean(axis=0)

    print(f"\nContact at index {contact_idx_og} ({contact_time_og:.2f} s)")
    print(f"Settles at index {settle_idx_og} ({settle_time_og:.2f} s)")
    print(f"Primary tipping axis is {TIP_AXIS.round(3)} (avg over window)")

## ================ Plot raw data ===================
    if plot_raw:
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax = plot_3vec(ax, time, f_exp_raw, label='Force (raw)', linew=2)
        ax = plot_3vec(ax, time, force_exp, label='Filt', linew=2)
        ax.axvline(contact_time_og, color='k', linestyle='--', linewidth=2, label='Contact & Settle')
        ax.axvline(settle_time_og, color='k', linestyle='--', linewidth=2, label='_')
        ax2 = plt.twinx()
        ax2 = plot_3vec(ax2, time, axes_exp_raw, label='Object axis (raw)', linew=3)
        ax2.plot(time, np.rad2deg(theta_exp), color='g', linestyle='-', linewidth=3, label='Object angle (raw)')
    
        set_fig_opts(ax, 'Time (s)', 'Force (N)', ax2, 'Angle (deg)')
        plt.title(f"Raw Force Data for {shape.upper()} Object", fontsize=15)
        fig.legend(loc='upper left', fontsize=15)
        # FOR PAPER FIGURE, TRIM THE AXES LIMITS
        # ax.set_xlim(1.6, 6)
        # ax2.set_ylim(0, 30)


    if plot_ee:
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax = plot_3vec(ax, time, ee_exp, label='EE Pos. (raw)', linew=3)
        ax.axvline(contact_time_og, color='k', linestyle='--', linewidth=2, label='_')
        ax.axvline(settle_time_og, color='k', linestyle='--', linewidth=2, label='_')
        ax2 = plt.twinx()
        ax2.plot(time, np.rad2deg(theta_exp), color='g', linestyle='-', linewidth=3, label='Object angle (raw)')
        set_fig_opts(ax, 'Time (s)', 'EE Position (m)', ax2, 'Angle (deg)')
        fig.legend(loc='upper left', fontsize=15)

        # FOR PAPER FIGURE, TRIM THE AXES LIMITS
        # ax.set_xlim(1.6, 6)
        # ax2.set_ylim(0, 30)

    ## =================== HACK: Modify experiment parameters to make better fitting ===================
    # # Don't currently have exact measure for o_obj. HOWEVER, at theta_crit, the EE x-pos should be equal to o x-pos!
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
    lin_slope, lin_b, _, _, _ = linregress(th_trim, f_trim[:, 0])
    theta_star_calc = -lin_b / lin_slope
    zc_calc = abs(abs(com_gt[0]) / np.tan(theta_star_calc))
    m_calc = abs(abs(lin_slope)*ee_trim[-1,2]/(9.81*zc_calc))

    print(f"\nRecall init guess from linear fit:")
    print(f"mass: {m_calc:.3f} kg (GT: {m_gt:.3f} kg)")
    print(f"zc: {zc_calc:.3f} m (GT: {com_gt[2]:.3f} m)")
    print(f"theta*: {np.rad2deg(theta_star_calc):.2f} deg (GT: {np.rad2deg(theta_star_gt):.2f} deg)")
    
    ## =============== Fit using TAU model ==================
    rc0_known       = np.array([com_gt[0], com_gt[1], 0.0])
    rf              = (ee_trim - o_obj)

    # Before fitting, must pre-compute corresponding PUSH torque
    f_app = -f_trim # NOTE: IMPORTANT NEGATE TO MATCH F OBJECT EXPERIENCES

    # TEMP HACK: FRICTION IN Z COMPONENT IS PARASITIC FOR TORQUE, SO ZERO IT OUT FOR NOW...
    f_app[:, 1] = 0.0
    f_app[:, 2] = 0.0
    tau_app_trim    = tau_app_model(f_app, rf)

    [m_est, zc_est], pcov = curve_fit(
        lambda th, m, zc: tau_model(th, m, zc, rc0_known=rc0_known, e_hat=TIP_AXIS),
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
    tau_model_est = tau_model(th_trim, m_est, zc_est, rc0_known=rc0_known, e_hat=TIP_AXIS).reshape(-1,3)
    # ax.plot(np.rad2deg(th_trim), tau_model_est[:,0], color='b', linestyle='-', linewidth=3, label='Model fit (experienced) X')
    ax.plot(np.rad2deg(th_trim), tau_model_est[:,1], color='b', linestyle='-', linewidth=3, label='Model fit (experienced) Y')
    # ax.plot(np.rad2deg(th_trim), tau_model_est[:,2], color='b', linestyle='-', linewidth=3, label='Model fit (experienced) Z')
    # Plot our model but to extrapolated full theta range
    tau_model_full_est = tau_model(th_extrap_est, m_est, zc_est, rc0_known=rc0_known, e_hat=TIP_AXIS).reshape(-1,3)
    # ax.scatter(np.rad2deg(th_extrap_est), tau_model_full_est[:,0], color='m', label='Full fit (est) X')
    ax.scatter(np.rad2deg(th_extrap_est), tau_model_full_est[:,1], color='m', label='Full fit (est) Y')
    # ax.scatter(np.rad2deg(th_extrap_est), tau_model_full_est[:,2], color='m', label='Full fit (est) Z')
    # Plot using the ground truth params for reference
    tau_model_full_gt = tau_model(th_extrap_est, m_gt, com_gt[2], rc0_known=rc0_known, e_hat=TIP_AXIS).reshape(-1,3)
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
    f_app_model_est = F_model(th_trim, m_est, zc_est, rf, rc0_known=rc0_known, e_hat=TIP_AXIS)
    # Extrapolate experienced data to mimic a full toppling experiment for plotting
    f_app_full_est = F_model(th_extrap_est, m_est, zc_est, rf, rc0_known=rc0_known, e_hat=TIP_AXIS)
    f_app_full_gt  = F_model(th_extrap_est, m_gt, com_gt[2], rf, rc0_known=rc0_known, e_hat=TIP_AXIS)

    ## IMPORTANT: must negate the 'experienced force' back again!
    f_app_model_est *= -1
    f_app_full_est  *= -1
    f_app_full_gt   *= -1

    f_plot_exp = f_trim[:,0]
    y_label = "X-Force (N)"
    f_plot_model_est = f_app_model_est[:,0]
    f_plot_full_est = f_app_full_est[:,0]
    f_plot_full_gt = f_app_full_gt[:,0]

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

    return m_est, zc_est, theta_star_est


if __name__ == "__main__":
    shapes_to_run = ["box", "heart", "flashlight"] #, "lshape"]#, "monitor"]

    for shape in shapes_to_run:
        obj     = OBJECTS[shape]
        path    = obj["path"]
        com     = obj["com"]
        m_gt    = obj["mass"]
        # Ground truth from geometry
        theta_star_gt = np.rad2deg(np.arctan2(abs(np.linalg.norm(com[0:1])), com[2]))
        OBJECTS[shape]["theta_star_gt"] = theta_star_gt

        print(f"\n========= Analyzing object: {shape.upper()} =========")
        print(f"Ground truth from geometry:\ntheta*: {theta_star_gt:.2f} deg, zc = {com[2]:.3f} m")

        m_est, zc_est, theta_star_est = main(shape, path, com, m_gt, theta_star_gt, plot_raw=True, plot_ee=False)

        obj["est"] = [m_est, zc_est, theta_star_est]


    print("\n\n========= FINAL ESTIMATES ========= ")
    for shape in shapes_to_run:
        obj             = OBJECTS[shape]
        com_gt          = obj["com"]
        m_gt            = obj["mass"]
        theta_star_gt   = obj["theta_star_gt"]
        height          = obj["height"]
        m_est, zc_est, theta_star_est = obj["est"]
        print(f"{shape.upper()}:")
        print(f"Estimated: mass = {m_est:.3f} kg, zc = {zc_est:.3f} m, theta* = {np.rad2deg(theta_star_est):.2f} deg")
        print(f"Gnd Truth: mass = {m_gt:.3f} kg, zc = {com[2]:.3f} m, theta* = {theta_star_gt:.2f} deg")
        
        # Error as a percentage of mass, object height, and theta*
        m_err = m_est - m_gt
        m_err_pct = abs(m_err)/m_gt * 100
        zc_err = zc_est - com[2]
        zc_err_pct = abs(zc_err)/height * 100
        theta_star_err = np.rad2deg(theta_star_est) - theta_star_gt
        theta_star_pct = abs(theta_star_err)/theta_star_gt * 100

        print(f"   Errors: mass = {m_err:.3f} kg, zc = {zc_err:.3f} m, theta* = {theta_star_err:.2f} deg")
        print(f"  Percent: mass = {m_err_pct:.3f} %,  zc = {zc_err_pct:.3f} %,  theta* = {theta_star_pct:.3f} %\n")

    plt.tight_layout()
    plt.show()