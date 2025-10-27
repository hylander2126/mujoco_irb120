import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from .helper_fns import axisangle2rot

## Theta model (input is theta, output is force)
def theta_model(x, a, b):
    F           = x.reshape(-1,3)  # ensure shape is (n,3)
    m           = a
    zc          = b
    
    g           = 9.81
    rf0         = np.array([-0.1, 0.0, 0.25])
    rc0_known   = np.array([-0.05, 0.0,  0.0])
    e_hat       = np.array([  0.0, 1.0,  0.0])
    z_hat       = np.array([  0.0, 0.0,  1.0])
    rc0         = rc0_known + np.array([0.0, 0.0, zc])

    a = np.cross(e_hat, rf0) # (3,)
    b = np.cross(e_hat, rc0) # (3,)
    # C = (a . F) + mg(b . z_hat) (but np shapes don't match. eF . a is equivalent but yield (n,) which is preferred)
    C = (F @ a) + m*g*(b @ z_hat)
    eF = np.cross(e_hat, F)
    eZ = np.cross(e_hat, z_hat)
    # D = (a . eF) + mg(b . eZ)
    D = (eF @ a) + m*g*(b @ eZ)

    return np.arctan2(C, D)  # (n,)

## Force model (input is theta, output is force)
def F_model(x, a, b, push_dirs):
    theta       = np.asarray(x).reshape(-1,)  # ensure shape is (n,)
    m           = a
    zc          = b

    g           = 9.81
    rf0         = np.array([-0.1, 0.0, 0.25])
    rc0_known   = np.array([-0.05, 0.0,  0.0])
    e_hat       = np.array([  0.0, 1.0,  0.0])
    z_hat       = np.array([  0.0, 0.0,  1.0])
    rc0         = rc0_known + np.array([0.0, 0.0, zc])

    # normalize push_dir
    push_dirs = np.asarray(push_dirs).reshape(-1,3)  # ensure shape is (N,3)
    d = push_dirs / np.linalg.norm(push_dirs, axis=1, keepdims=True) # (N,3)

    # precompute some pieces
    e_cross_rf0  = np.cross(e_hat, rf0)        # (3,)
    z_cross_ehat = np.cross(z_hat, e_hat)      # (3,)

    R_pos = axisangle2rot(e_hat, theta)   # (N,3,3)
    R_neg = axisangle2rot(e_hat, -theta)  # (N,3,3)

    # A(theta)
    A = (R_pos @ e_cross_rf0.reshape(3,1)).reshape(-1,3)  # (N,3)

    # tmp(theta)
    tmp = (R_neg @ z_cross_ehat.reshape(3,1)).reshape(-1,3)  # (N,3)

    # B(theta)
    # B = m g * rc0^T * tmp
    B = m * g * (tmp @ rc0.reshape(3,1)).reshape(-1)  # (N,)

    # Now solve for alpha per theta:
    # alpha = B / (A^T d)
    denom = np.einsum('ij,ij->i', A, d)  # (N,), dot(A[i], d)
    alpha = B / denom                   # (N,)

    # F(theta) = alpha * d
    F_pred = alpha[:,None] * d # [None,:]  # (N,3)

    print(a, b)
    
    return F_pred

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