import numpy as np
import time


def as_numpy(value):
    """Convert Genesis/Torch values to plain NumPy arrays for controller math."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def trapezoid_speed(step, ramp_up_steps, hold_steps, ramp_down_steps, max_speed):
    """
    Scalar speed command for a trapezoidal velocity profile.

    The profile is intentionally scalar: callers multiply the result by a
    Cartesian direction vector. That makes the same profile useful for pushes
    along x, y, z, or any normalized 3D direction.
    """
    total_steps = ramp_up_steps + hold_steps + ramp_down_steps

    if step < 0 or step >= total_steps:
        return 0.0

    if step < ramp_up_steps:
        return max_speed * (step + 1) / ramp_up_steps

    if step < ramp_up_steps + hold_steps:
        return max_speed

    ramp_down_step = step - ramp_up_steps - hold_steps
    return max_speed * (1.0 - (ramp_down_step + 1) / ramp_down_steps)


def smoothstep(value):
    """
    Smoothly map [0, 1] -> [0, 1] with zero slope at both ends.

    This avoids a sharp velocity kink when safety checks start scaling the
    Cartesian command near workspace or manipulability limits.
    """
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def ellipsoid_speed_scale(point, center, radii, core_fraction=0.80):
    """
    Cheap workspace guard.

    The outer allowed workspace is:
        ((x-cx)/rx)^2 + ((y-cy)/ry)^2 + ((z-cz)/rz)^2 <= 1

    The inner core allows full speed. Between the core and the outer boundary,
    speed fades smoothly to zero. Outside the ellipsoid, motion stops.
    """
    point = np.asarray(point, dtype=float)
    center = np.asarray(center, dtype=float)
    radii = np.asarray(radii, dtype=float)

    ellipsoid_value = np.sum(((point - center) / radii) ** 2)
    core_value = core_fraction ** 2

    if ellipsoid_value <= core_value:
        return 1.0
    if ellipsoid_value >= 1.0:
        return 0.0

    boundary_fraction = (1.0 - ellipsoid_value) / (1.0 - core_value)
    return smoothstep(boundary_fraction)


def manipulability_speed_scale(jac_pos, stop_sigma=0.01, full_speed_sigma=0.05):
    """
    Cheap singularity/manipulability guard using the translational Jacobian.

    The smallest singular value measures how close the translational velocity
    solve is to becoming ill-conditioned. As it approaches zero, the same
    Cartesian velocity requires larger and larger joint velocities, so we fade
    the commanded speed out before the pseudo-inverse gets exciting.
    """
    singular_values = np.linalg.svd(jac_pos, compute_uv=False)
    sigma_min = singular_values[-1]

    if sigma_min <= stop_sigma:
        return 0.0
    if sigma_min >= full_speed_sigma:
        return 1.0

    return smoothstep((sigma_min - stop_sigma) / (full_speed_sigma - stop_sigma))


def quat_rotate(quat, vector):
    """Rotate a vector by a Genesis/MuJoCo-style quaternion ordered [w, x, y, z]."""
    w, x, y, z = quat
    q_vec = np.array([x, y, z])
    vector = np.asarray(vector, dtype=float)
    return vector + 2.0 * np.cross(q_vec, np.cross(q_vec, vector) + w * vector)


def damped_least_squares_qdot(jac, target_velocity, damping=0.01):
    """
    Solve J qdot = target_velocity with damped least squares.

    Damping keeps the solve finite near singularities. The manipulability guard
    should reduce speed before this becomes extreme, but damping is a useful
    second layer of protection.
    """
    lhs = jac @ jac.T + (damping ** 2) * np.eye(jac.shape[0])
    return jac.T @ np.linalg.solve(lhs, target_velocity)


def damped_pseudoinverse(jac, damping=0.01):
    """
    Return the damped least-squares pseudo-inverse J#.

    Keeping this separate from damped_least_squares_qdot lets us reuse J# for
    nullspace projection: qdot = J# v + (I - J#J) qdot_posture.
    """
    lhs = jac @ jac.T + (damping ** 2) * np.eye(jac.shape[0])
    return jac.T @ np.linalg.solve(lhs, np.eye(jac.shape[0]))


def limit_joint_velocity(qdot, limit):
    """
    Preserve the solved joint-velocity direction while respecting a max speed.

    Per-joint clipping changes the direction of qdot, which can destroy the
    Cartesian velocity solve. Uniform scaling keeps the same qdot direction and
    only reduces its magnitude when one joint would exceed the limit.
    """
    max_abs = np.max(np.abs(qdot))
    if max_abs <= limit:
        return qdot, 1.0
    scale = limit / max_abs
    return qdot * scale, scale


class GenesisRobotController:
    """
    Small Genesis-side controller wrapper for the IRB120.

    Genesis already gives us IK, path planning, and joint control APIs, so this
    class mainly owns project-specific details: joint names, default gains,
    contact-point Jacobian control, trapezoidal push profiles, workspace fade
    out, and manipulability fade out.
    """

    def __init__(self, entity, scene, joint_names=None):
        self.entity = entity
        self.scene = scene
        self.joint_names = joint_names or [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
        self.dofs_idx = [
            idx
            for name in self.joint_names
            for idx in self.entity.get_joint(name).dofs_idx_local
        ]

        self.pusher = self.entity.get_link("pusher_link")

        # Near the center of the pusher ball in pusher_link coordinates.
        # Use this point for the Jacobian so "push straight" means the contact
        # point moves straight, not merely the pusher link origin.
        self.contact_local_point = np.array([0.09675, 0.0, 0.0])

        # Tune these to define the task-space region where pushes are allowed.
        self.workspace_center = np.array([0.32, 0.0, 0.30])
        self.workspace_radii = np.array([0.42, 0.45, 0.32])
        self.workspace_core_fraction = 0.80

        # Singularity guard thresholds for the translational Jacobian.
        self.stop_sigma = 0.01
        self.full_speed_sigma = 0.05

        # Last-resort joint velocity cap. This catches pseudo-inverse spikes
        # and keeps the commanded joint velocities sane near awkward poses.
        self.joint_velocity_limit = 1.5

    def configure_default_gains(self):
        """Apply the default gains/force ranges used by the Genesis smoke test."""
        self.entity.set_dofs_kp(
            kp=np.array([3500, 3500, 2500, 2500, 1000, 1000]),
            dofs_idx_local=self.dofs_idx,
        )
        self.entity.set_dofs_kv(
            kv=np.array([350, 350, 250, 250, 100, 100]),
            dofs_idx_local=self.dofs_idx,
        )
        self.entity.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12]),
            upper=np.array([87, 87, 87, 87, 12, 12]),
            dofs_idx_local=self.dofs_idx,
        )

    def step(self, camera=None):
        """
        Step the scene and optionally render one camera frame.

        Genesis camera recording stores frames produced by camera.render(), so
        headless video capture needs an explicit render after each sim step.
        """
        self.scene.step()
        if camera is not None:
            camera.render()

    def link_local_point_world(self, link=None, local_point=None):
        """Convert a point fixed in a link's local frame into world coordinates."""
        link = link or self.pusher
        local_point = self.contact_local_point if local_point is None else local_point
        link_pos = as_numpy(link.get_pos())
        link_quat = as_numpy(link.get_quat())
        return link_pos + quat_rotate(link_quat, local_point)

    def get_contact_jacobian(self):
        """
        Return the 6D Jacobian at the pusher contact point.

        Genesis row order is [vx, vy, vz, wx, wy, wz]. Most pushing control only
        needs rows 0:3, which correspond to translational velocity of this point.
        """
        return as_numpy(
            self.entity.get_jacobian(
                self.pusher,
                local_point=self.contact_local_point,
            )
        )

    def plan_ik_with_constraints(self, pos, quat, t_const=5, link=None, camera=None):
        """
        Use Genesis IK and RRTConnect planning to reach a Cartesian pose.

        This intentionally keeps your existing planner-based preshove behavior:
        the method executes each waypoint in the scene and returns the planned
        path for the caller to inspect or reuse.
        """
        link = link or self.pusher
        qpos = self.entity.inverse_kinematics(link=link, pos=pos, quat=quat)

        path = self.entity.plan_path(
            qpos_goal=qpos,
            qpos_start=None,
            planner="RRTConnect",
            num_waypoints=t_const * 100,
            resolution=0.05,
            smooth_path=True,
            max_nodes=4000,
            timeout=None,
            max_retry=1,
            ignore_collision=False,
            with_entity=None,
        )
        if path is None:
            print("Failed to plan a valid path.")
            return None

        for waypoint in path:
            # self.entity.set_dofs_position(waypoint, self.dofs_idx)
            self.entity.control_dofs_position(waypoint, self.dofs_idx)
            self.step(camera=camera)

        return path

    def stop_velocity(self, steps=500, camera=None):
        """Command zero joint velocity for a few steps so the robot settles."""
        zeros = np.zeros(len(self.dofs_idx))
        for _ in range(steps):
            self.entity.control_dofs_velocity(zeros, self.dofs_idx)
            self.step(camera=camera)

    def velocity_shove(
        self,
        shove_speed=0.55,
        preshove_pos=None,
        preshove_quat=None,
        push_direction=None,
        obj=None,
        camera=None,
        ramp_up_steps=25,
        hold_steps=50,
        ramp_down_steps=25,
        settle_steps=100,
        snap=False,
        height_kp=4.0,
        posture_kp=0.75,
    ):
        """
        Execute a guarded Cartesian velocity shove.

        The command pipeline is:
            trapezoid speed
            * workspace ellipsoid scale
            * manipulability scale
            -> damped least-squares joint velocity
            -> joint velocity clamp
        """
        # z=0.18 is low, z=0.25 is centroid, z=0.3 is top
        preshove_pos = np.array([0.30, 0.0, 0.25]) if preshove_pos is None else np.asarray(preshove_pos, dtype=float)
        preshove_quat = np.array([1, 0, 0, 0]) if preshove_quat is None else np.asarray(preshove_quat, dtype=float)
        push_direction = np.array([1.0, 0.0, 0.0]) if push_direction is None else np.asarray(push_direction, dtype=float)
        push_direction = push_direction / np.linalg.norm(push_direction)

        timeout = 25.0
        start_time = time.time()

        if snap:
            # Snap to preshove pose, but first convert cartesian to joint angles
            q_preshove = self.entity.inverse_kinematics(link=self.pusher, pos=preshove_pos, quat=preshove_quat)
            self.entity.set_dofs_position(q_preshove, self.dofs_idx)
            
        else:
            q_preshove_plan = self.plan_ik_with_constraints(
                pos=preshove_pos,
                quat=preshove_quat,
                t_const=5,
                camera=camera,
            )
            if q_preshove_plan is None:
                return

            q_preshove = q_preshove_plan[-1]

        q_preshove = as_numpy(q_preshove)

        # Briefly hold the preshove pose before making contact.
        for _ in range(100):
            self.entity.control_dofs_position(q_preshove, self.dofs_idx)
            self.step(camera=camera)

        # Hold the contact point's vertical position from the start of the
        # shove. Lateral y motion is left unconstrained on purpose.
        contact_ref = self.link_local_point_world()
        prev_contact_pos = contact_ref.copy()

        shove_steps = ramp_up_steps + hold_steps + ramp_down_steps
        for step in range(shove_steps):
            contact_pos = self.link_local_point_world()
            jac = self.get_contact_jacobian()
            jac_pos = jac[:3, :]

            profile_speed = trapezoid_speed(
                step,
                ramp_up_steps,
                hold_steps,
                ramp_down_steps,
                shove_speed,
            )
            workspace_scale = ellipsoid_speed_scale(
                contact_pos,
                self.workspace_center,
                self.workspace_radii,
                core_fraction=self.workspace_core_fraction,
            )
            manip_scale = manipulability_speed_scale(
                jac_pos,
                stop_sigma=self.stop_sigma,
                full_speed_sigma=self.full_speed_sigma,
            )
            commanded_speed = profile_speed * workspace_scale * manip_scale

            if profile_speed <= 1e-6:
                break

            if workspace_scale <= 1e-6 or manip_scale <= 1e-6:
                print(
                    "Stopping shove:",
                    f"profile={profile_speed:.3f}",
                    f"workspace_scale={workspace_scale:.3f}",
                    f"manip_scale={manip_scale:.3f}",
                    f"contact_pos={contact_pos}",
                )
                break

            if (time.time() - start_time) > timeout:
                print("Shove timeout reached.")
                break

            # Feedforward shove speed plus feedback to keep y/z near the
            # preshove contact point. For the default +x shove, this gives a
            # constant-height Cartesian command without constraining lateral y.
            target_velocity = commanded_speed * push_direction
            target_velocity[2] += height_kp * (contact_ref[2] - contact_pos[2])

            jac_pinv = damped_pseudoinverse(jac_pos, damping=0.01)
            qdot_task = jac_pinv @ target_velocity

            # The translational task only uses 3 constraints for a 6-DOF arm.
            # This nullspace term asks the unused DOFs to stay near the
            # preshove posture instead of drifting into odd elbow/wrist poses.
            q_current = as_numpy(self.entity.get_dofs_position(self.dofs_idx))
            qdot_posture = posture_kp * (q_preshove - q_current)
            nullspace = np.eye(len(self.dofs_idx)) - jac_pinv @ jac_pos
            qdot_unlimited = qdot_task + nullspace @ qdot_posture

            qdot, qdot_scale = limit_joint_velocity(qdot_unlimited, self.joint_velocity_limit)
            predicted_velocity = jac_pos @ qdot

            self.entity.control_dofs_velocity(qdot, self.dofs_idx)
            self.step(camera=camera)

            new_contact_pos = self.link_local_point_world()
            actual_delta = new_contact_pos - prev_contact_pos
            prev_contact_pos = new_contact_pos

            if step % 100 == 0:
                box_pos = as_numpy(obj.get_pos()) if obj is not None else None
                print(
                    "Shove:",
                    f"speed={commanded_speed:.3f}",
                    f"workspace_scale={workspace_scale:.3f}",
                    f"manip_scale={manip_scale:.3f}",
                    f"box_pos={box_pos}",
                    f"fingertip_pos={contact_pos}",
                    f"target_v={target_velocity}",
                    f"predicted_v={predicted_velocity}",
                    f"actual_delta={actual_delta}",
                    f"qdot_scale={qdot_scale:.3f}",
                    f"qdot={qdot}",
                )

        self.stop_velocity(steps=settle_steps, camera=camera)
