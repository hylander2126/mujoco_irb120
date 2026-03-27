import numpy as np
from .helper_fns import axisangle2rot, rotvec_to_rot, vec_to_unit, VecToso3

def get_adT_sensor_O(rot_vecs, p_sensor_B, p_obj_B):
    """
    Compute adjoint transform from object frame to sensor frame, given object rotation and sensor position.

    {O}, {B}, {S} are object, robot base/table/world, and sensor frames, respectively.

    rot_vecs: (N,3) array of axis-angle rotation vectors (angle in radians)
    p_sensor_B: (3,) position of sensor in robot base frame
    p_obj_B: (3,) position of object in robot base frame
    """
    R_O = rotvec_to_rot(rot_vecs)  # (N,3,3) Object rotation in sensor frame
    R_O_T = R_O.transpose(0, 2, 1)  # (N,3,3) Transpose for inverse rotation (swaps correctly each 3x3 block)
    adT_S_O = np.zeros((rot_vecs.shape[0], 6, 6))
    adT_S_O[:, :3, :3] = R_O
    adT_S_O[:, 3:, 3:] = R_O
    # Calculate coordinates of sensor frame IN OBJECT FRAME:
    p_S_O = R_O_T @ (p_sensor_B - p_obj_B)
    adT_S_O[:, :3, 3:] = -R_O @ VecToso3(p_S_O)
    return adT_S_O

def model_bkwd_wrench(
    w_meas_S: np.ndarray,
    adT_sensor_O: np.ndarray,
    p_finger_O: np.ndarray,
):
    """
    Compute 'backward' applied wrench [F; tau] IN OBJECT FRAME
    {O}, {B}, {S} are object, robot base/table/world, and sensor frames, respectively.

    w_meas_S: (N,6) array of measured wrenches in sensor frame (F_x, F_y, F_z, tau_x, tau_y, tau_z)
    adT_sensor_O: (N,4,4) array of adjoint transforms from object frame to sensor frame
    p_finger_O: (3,) position of finger in object frame
    """
    w_meas_O = adT_sensor_O @ w_meas_S.T  # (N,6) measured wrench in object frame

    ## CONSTRUCT APPLIED WRENCH IN OBJECT FRAME
    f_app_O = -w_meas_O[:3].T
    t_app_O = np.cross(p_finger_O, f_app_O)  # Torque applied ON object by finger is r_o_to_finger x f_app
    w_app_O = np.hstack((f_app_O, t_app_O))  # (N,6) applied wrench on object frame
    return w_app_O


def model_fwd_wrench(
        rot_vecs: np.ndarray,
        p_c_O: np.ndarray,
        mass: float,
        mu_table: float,
        N_table: float,
        rob_vel_B: np.ndarray = None,
):
    """
    Compute 'forward' gravity + ground reaction wrench [F; tau] IN OBJECT FRAME
    {O}, {B}, {S} are object, robot base/table/world, and sensor frames, respectively.

    rot_vecs: (N,3) array of axis-angle rotation vectors (angle in radians)
    w_app_O: (N,6) array of applied wrenches in object frame (F_x, F_y, F_z, tau_x, tau_y, tau_z)
    adT_sensor_O: (N,4,4) array of adjoint transforms from object frame to sensor frame
    rob_vel_B: (N,3) array of robot/finger velocities in robot base/world frame (v_x, v_y, v_z)

    p_c_O: (3,) position of object CoM in object frame
    mass: scalar mass of the object
    mu_table: scalar friction coefficient of the table
    N_table: scalar normal force magnitude from the table
    """
    rot_vecs = np.asarray(rot_vecs, dtype=float)
    if rot_vecs.ndim != 2 or rot_vecs.shape[1] != 3:
        raise ValueError(f"rot_vecs must have shape (N,3), got {rot_vecs.shape}")

    R_B = rotvec_to_rot(rot_vecs)  # (N,3,3) object rotation in world frame
    R_B_T = R_B.transpose(0, 2, 1)  # (N,3b,3a) Transpose for inverse rotation (swaps correctly each 3x3 block)
    g_B = np.array([0, 0, -9.81])  # gravity in world/robot/table frame
    if rob_vel_B is None:
        rob_vel_B = np.tile(np.array([-1.0, 0.0, 0.0]), (rot_vecs.shape[0], 1))
    else:
        rob_vel_B = np.asarray(rob_vel_B, dtype=float)
        if rob_vel_B.shape != (rot_vecs.shape[0], 3):
            # Check if rot_vecs is 1,3, if so, pass
            if rot_vecs.shape[0] == 1 and rob_vel_B.shape == (3,):
                rob_vel_B = rob_vel_B.reshape(1, 3)
            else:
                raise ValueError(
                    f"rob_vel_B must have shape {(rot_vecs.shape[0], 3)}, got {rob_vel_B.shape}"
                )

    ## CONSTRUCT GRAVITY WRENCH IN OBJECT FRAME
    g_O = R_B_T @ g_B                               # (N,3) gravity in object frame
    f_grav_O = mass * g_O                           # (N,3) gravity force in object frame
    tau_grav_O = np.cross(p_c_O, f_grav_O)          # (N,3) gravity torque in object frame about CoM
    w_grav_O = np.hstack((f_grav_O, tau_grav_O))    # (N,6) gravity wrench in object frame

    ## CONSTRUCT GROUND REACTION WRENCH IN OBJECT FRAME # NOTE: currently assumed constant table friction
    # Row-wise unit direction. If zero speed, set friction to zero for that row.
    vel_norm = np.linalg.norm(rob_vel_B, axis=1, keepdims=True)
    dir_S = np.zeros_like(rob_vel_B)
    moving = vel_norm[:, 0] > 1e-12
    dir_S[moving] = rob_vel_B[moving] / vel_norm[moving]

    f_fr_S = -dir_S * (mu_table * N_table)  # (N,3) friction force in sensor/world frame
    # 'nij' is A (N,3,3) 'nj' is B (N,3) -> 'ni' is (N,3)  ---> 'n' iterated for mult. 'j' matches dot prod rules, 'i' is remaining axis
    f_fr_O = np.einsum('nij,ni->ni', R_B_T, f_fr_S)  # (N,3) friction force in object frame
    tau_ground_O = np.zeros_like(f_fr_O)               # (N,3) zero ground reaction torque about pivot (tipping edge)
    w_ground_O = np.hstack((f_fr_O, tau_ground_O))       # (N,6) ground reaction wrench in object frame

    # print("\nGravity wrench in object frame:\n", w_grav_O)
    print("Ground reaction wrench in object frame:\n", w_ground_O)
    
    return w_grav_O + w_ground_O

# ============================================================================== #
# ========================= OLD MODELS  ========================= #
# ============================================================================== #

def tau_app_model(F, rf):
    """
    Compute torque about pivot due to applied force F at position rf.

    rf must be same shape as F (N, 3) and must account for object rotation.
    """
    # return np.cross(F, rf)
    tau = np.cross(rf, F)  # (N,3)
    return tau.ravel()


def tau_model(theta, m, zc, rc0_known, e_hat=[0,1,0]):
    """
    Compute the gravity torque given theta, mass, and z-height of CoM
    """
    W           = np.array([0, 0, -9.8067 * m]) # Weight in space frame
    # rc0_known   = np.array([-0.05, 0.0,  0.0]) # -0.05 , 0 , 0
    e_hat       = np.asarray(e_hat).flatten()  # ensure shape is (3,)
    rc0         = rc0_known.copy()
    rc0[2]      = zc
    theta       = np.asarray(theta).flatten()  # ensure shape is (n,)

    # TEMP testing new strategy
    # Get (batch) rotation matrix from axis-angle
    # -(rc0 x R(-theta)W)
    R = axisangle2rot(e_hat, -theta)   # (N,3,3)

    W_rotated = R @ W
    tau = -np.cross(rc0, W_rotated)  # (N,3)
    return tau.ravel()

## Force model (input is theta, output is force)
def F_model(theta, m, zc, rf, rc0_known, e_hat=[0,1,0]):
    """
    Force model: given angle(s) theta, mass m, CoM height zc, and
    per-sample lever arm rf (N,3) in the object frame, return the
    predicted contact force F(theta) in the object frame (N,3).

    theta : array-like, shape (N,) or (N,1)
    m     : mass
    zc    : CoM height above rc0_known.z
    rf    : lever arm from pivot to finger contact, shape (N,3)
    """
    theta = np.asarray(theta).reshape(-1)   # (N,)
    rf    = np.asarray(rf)                  # (N,3)
    N     = theta.shape[0]
    assert rf.shape == (N, 3), "rf must have shape (N,3)"

    g = 9.81
    # Geometry / axes in object frame
    e_hat     = np.asarray(e_hat).flatten()  # ensure shape is (3,)
    z_hat     = np.array([ 0.0, 0.0, 1.0])    # world/object z

    # CoM at height zc above rc0_known in z-direction
    rc0 = rc0_known.copy()
    rc0[2] = zc   # (3,)

    # 👉 Push direction in object frame (assumed constant)
    # Change to +1.0 if you push in +x in the object frame.
    d_hat = np.array([1.0, 0.0, 0.0])          # (3,)

    # Rotation matrices around e_hat by +theta and -theta
    R_pos = axisangle2rot(e_hat,  theta)        # (N,3,3)
    R_neg = axisangle2rot(e_hat, -theta)        # (N,3,3)

    # A(theta) = R_pos * (e × r_f)
    e_cross_rf = np.cross(e_hat, rf)            # (N,3)
    A = np.einsum('nij,nj->ni', R_pos, e_cross_rf)   # (N,3)

    # tmp(theta) = R_neg * (z × e)
    z_cross_ehat = np.cross(z_hat, e_hat)       # (3,)
    tmp = np.einsum('nij,j->ni', R_neg, z_cross_ehat)  # (N,3)

    # B(theta) = m g rc0ᵀ tmp  → (N,)
    B = m * g * (tmp @ rc0)

    # denom = Aᵀ d_hat = dot(A[i], d_hat), shape (N,)
    denom = A @ d_hat

    # alpha(theta) = B / (Aᵀ d_hat)
    alpha = B / denom                          # (N,)

    # F(theta) = alpha * d_hat  → (N,3)
    F_pred = alpha[:, None] * d_hat            # (N,3)

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