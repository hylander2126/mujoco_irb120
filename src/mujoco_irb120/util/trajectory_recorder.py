"""
Trajectory Recording and Playback Module

A standalone module for recording and replaying robot trajectories with waypoint-based control.
Supports multiple recording types (joint angles, end-effector poses, cartesian positions) and 
flexible playback strategies.

Usage:
    from mujoco_irb120.common.trajectory_recorder import TrajectoryRecorder
    
    recorder = TrajectoryRecorder(robot_controller)
    recorder.start_recording()
    # ... run simulation ...
    recorder.stop_recording()
    recorder.save_trajectory("my_traj.npz")
"""

import numpy as np
import json
from pathlib import Path


class TrajectoryRecorder:
    """Manages recording, playback, and storage of robot trajectories."""
    
    def __init__(self, robot_controller):
        """Initialize the trajectory recorder.
        
        Args:
            robot_controller: Instance of robot_controller.controller
        """
        self.robot = robot_controller
        self.recording_enabled = False
        self.playback_enabled = False
        self.record_forces = False
        self.waypoints = []
        self.waypoint_times = []
        self.forces = []  # Recorded force/wrench data
        self.playback_index = 0
        self.playback_start_time = 0.0
        self.playback_traj = None
        
    # ==================== RECORDING METHODS ====================
    
    def start_recording(self, verbose=True, record_forces=False):
        """Start recording trajectory waypoints.
        
        Clears any previous recording and begins capturing waypoints.
        
        Args:
            verbose (bool): Print status messages
            record_forces (bool): Also record F/T sensor readings (wrench)
        """
        self.recording_enabled = True
        self.playback_enabled = False
        self.record_forces = record_forces
        self.waypoints = []
        self.waypoint_times = []
        self.forces = []
        self.playback_index = 0
        msg = "[RECORDER] Started recording trajectory waypoints"
        if record_forces:
            msg += " (with force tracking)"
        if verbose:
            print(msg + ".")
    
    def stop_recording(self, verbose=True):
        """Stop recording trajectory waypoints.
        
        Args:
            verbose (bool): Print status messages
            
        Returns:
            dict: Summary of recorded trajectory
        """
        self.recording_enabled = False
        num_waypoints = len(self.waypoints)
        if num_waypoints > 0:
            duration = self.waypoint_times[-1] - self.waypoint_times[0]
            if verbose:
                print(f"[RECORDER] Stopped recording. Captured {num_waypoints} waypoints over {duration:.2f} seconds.")
            return {
                'num_waypoints': num_waypoints,
                'duration': duration,
                'waypoints': self.waypoints.copy(),
                'times': self.waypoint_times.copy()
            }
        else:
            if verbose:
                print("[RECORDER] Stopped recording. No waypoints captured.")
            return None
    
    def record_waypoint(self, record_type='joints'):
        """Record a single waypoint at the current time.
        
        Args:
            record_type (str): Type of data to record. Options:
                - 'joints': Record joint positions (q)
                - 'pose': Record end-effector pose (4x4 transform)
                - 'cartesian': Record end-effector position only (3,)
        
        Returns:
            bool: True if waypoint was recorded, False otherwise
        """
        if not self.recording_enabled:
            return False
        
        current_time = self.robot.data.time
        
        if record_type == 'joints':
            waypoint = self.robot.data.qpos[self.robot.joint_idx].copy()
        elif record_type == 'pose':
            waypoint = self.robot.FK().copy()
        elif record_type == 'cartesian':
            waypoint = self.robot.FK()[:3, 3].copy()
        else:
            raise ValueError(f"Unknown record_type: {record_type}")
        
        self.waypoints.append(waypoint)
        self.waypoint_times.append(current_time)
        
        # Record force if enabled
        if self.record_forces:
            wrench = self.robot.ft_get_reading()  # Returns 6-vector: [fx, fy, fz, tx, ty, tz]
            self.forces.append(wrench.copy())
        
        return True
    
    def get_recorded_trajectory(self):
        """Get the currently recorded trajectory.
        
        Returns:
            dict: Dictionary with keys:
                - 'waypoints': list of waypoints
                - 'times': list of timestamps
                - 'num_waypoints': number of waypoints
                - 'duration': total trajectory duration (s)
                - 'forces': list of forces (if recorded), otherwise None
        """
        if len(self.waypoints) == 0:
            return None
        
        return {
            'waypoints': self.waypoints.copy(),
            'times': np.asarray(self.waypoint_times).copy(),
            'num_waypoints': len(self.waypoints),
            'duration': self.waypoint_times[-1] - self.waypoint_times[0],
            'forces': np.asarray(self.forces) if self.forces else None
        }
    
    def clear_trajectory(self, verbose=True):
        """Clear the current recorded trajectory.
        
        Args:
            verbose (bool): Print status message
        """
        num_cleared = len(self.waypoints)
        self.waypoints = []
        self.waypoint_times = []
        self.playback_index = 0
        self.recording_enabled = False
        self.playback_enabled = False
        if verbose and num_cleared > 0:
            print(f"[RECORDER] Cleared {num_cleared} waypoints.")
    
    # ==================== PLAYBACK METHODS ====================
    
    def start_playback(self, trajectory=None, verbose=True):
        """Start playback of a recorded trajectory.
        
        Args:
            trajectory (dict, optional): Pre-recorded trajectory dict. If None, uses current recording.
            verbose (bool): Print status messages
            
        Returns:
            bool: True if playback started successfully, False otherwise
        """
        if trajectory is None:
            trajectory = self.get_recorded_trajectory()
        
        if trajectory is None or len(trajectory.get('waypoints', [])) == 0:
            if verbose:
                print("[RECORDER] No trajectory available for playback.")
            return False
        
        self.recording_enabled = False
        self.playback_enabled = True
        self.playback_index = 0
        self.playback_start_time = self.robot.data.time
        self.playback_traj = trajectory
        
        if verbose:
            print(f"[RECORDER] Started playback of {trajectory['num_waypoints']} waypoints "
                  f"over {trajectory['duration']:.2f} seconds.")
        return True
    
    def stop_playback(self, verbose=True):
        """Stop playback of trajectory.
        
        Args:
            verbose (bool): Print status messages
        """
        self.playback_enabled = False
        self.playback_index = 0
        if verbose:
            print("[RECORDER] Stopped trajectory playback.")
    
    def playback_step(self, playback_type='joints', interpolate=True, maintain_orientation=False):
        """Execute one step of trajectory playback.
        
        Should be called every simulation step when playback_enabled is True.
        
        Args:
            playback_type (str): Type of playback control. Options:
                - 'joints': Direct joint position control
                - 'cartesian': Cartesian position control with IK
            interpolate (bool): Interpolate between waypoints or use nearest
            maintain_orientation (bool): Maintain orientation during cartesian playback
        
        Returns:
            bool: True if playback is still active, False if playback finished
        """
        if not self.playback_enabled:
            return False
        
        trajectory = self.playback_traj
        waypoints = trajectory['waypoints']
        times = trajectory['times']
        forces = trajectory.get('forces')
        
        elapsed_time = self.robot.data.time - self.playback_start_time
        
        # Check if we've reached the end of the trajectory
        if elapsed_time > times[-1] - times[0]:
            self.stop_playback(verbose=False)
            return False
        
        if interpolate:
            # Find the two waypoints to interpolate between
            target_time = times[0] + elapsed_time
            
            # Find the right index
            idx = np.searchsorted(times, target_time) - 1
            idx = np.clip(idx, 0, len(waypoints) - 2)
            
            t0, t1 = times[idx], times[idx + 1]
            w0, w1 = waypoints[idx], waypoints[idx + 1]
            
            # Interpolation factor
            alpha = (target_time - t0) / (t1 - t0) if t1 != t0 else 0.0
            alpha = np.clip(alpha, 0, 1)
            
            # Linear interpolation
            waypoint_interp = (1 - alpha) * w0 + alpha * w1
            
            # Interpolate force if available
            force_interp = None
            if forces is not None and len(forces) > 0:
                f0, f1 = forces[idx], forces[idx + 1]
                force_interp = (1 - alpha) * f0 + alpha * f1
        else:
            # Use nearest waypoint
            idx = np.argmin(np.abs(np.asarray(times) - (times[0] + elapsed_time)))
            waypoint_interp = waypoints[idx].copy()
            force_interp = forces[idx].copy() if forces is not None else None
        
        # Apply the waypoint based on playback type
        if playback_type == 'joints':
            self.robot.set_pos_ctrl(waypoint_interp, check_ellipsoid=False)
        elif playback_type == 'cartesian':
            # waypoint_interp should be a 4x4 transform
            if waypoint_interp.shape == (4, 4):
                try:
                    q_des = self.robot.IK(waypoint_interp, method=2, damping=0.5, max_iters=100)
                    self.robot.set_pos_ctrl(q_des, check_ellipsoid=False)
                except Exception as e:
                    print(f"[PLAYBACK] IK failed: {str(e)[:60]}")
                    return False
        
        # Apply force if forces were recorded (experimental force feedback)
        if force_interp is not None and hasattr(self.robot, 'ft_apply_wrench'):
            try:
                self.robot.ft_apply_wrench(force_interp)
            except Exception as e:
                pass  # Silently skip if force application not supported
        
        return True
    
    # ==================== FILE I/O METHODS ====================
    
    def save_trajectory(self, filepath, format='numpy'):
        """Save the current recorded trajectory to disk.
        
        Args:
            filepath (str): Path to save the trajectory
            format (str): Format to save. Options: 'numpy', 'json'
            
        Returns:
            bool: True if save successful, False otherwise
        """
        trajectory = self.get_recorded_trajectory()
        if trajectory is None:
            print("[RECORDER] No trajectory to save.")
            return False
        
        try:
            filepath = Path(filepath)
            
            if format == 'numpy':
                save_dict = {
                    'waypoints': np.asarray(trajectory['waypoints']),
                    'times': trajectory['times'],
                    'num_waypoints': trajectory['num_waypoints'],
                    'duration': trajectory['duration']
                }
                if trajectory['forces'] is not None:
                    save_dict['forces'] = trajectory['forces']
                np.savez(filepath, **save_dict)
            elif format == 'json':
                # Convert numpy arrays to lists for JSON serialization
                data = {
                    'num_waypoints': int(trajectory['num_waypoints']),
                    'duration': float(trajectory['duration']),
                    'waypoints': [w.tolist() for w in trajectory['waypoints']],
                    'times': trajectory['times'].tolist()
                }
                if trajectory['forces'] is not None:
                    data['forces'] = trajectory['forces'].tolist()
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unknown format: {format}")
            
            msg = f"[RECORDER] Saved trajectory to {filepath} ({format} format)"
            if trajectory['forces'] is not None:
                msg += " with force data"
            print(msg)
            return True
        
        except Exception as e:
            print(f"[RECORDER] Failed to save trajectory: {str(e)}")
            return False
    
    def load_trajectory(self, filepath, format='numpy'):
        """Load a previously saved trajectory from disk.
        
        Args:
            filepath (str): Path to load the trajectory from
            format (str): Format of the file. Options: 'numpy', 'json'
            
        Returns:
            dict: Loaded trajectory, or None if load failed
        """
        try:
            filepath = Path(filepath)
            
            if format == 'numpy':
                data = np.load(filepath, allow_pickle=True)
                trajectory = {
                    'waypoints': [data['waypoints'][i] for i in range(len(data['waypoints']))],
                    'times': data['times'],
                    'num_waypoints': int(data['num_waypoints']),
                    'duration': float(data['duration']),
                    'forces': data['forces'] if 'forces' in data else None
                }
            elif format == 'json':
                with open(filepath, 'r') as f:
                    data = json.load(f)
                trajectory = {
                    'waypoints': [np.asarray(w) for w in data['waypoints']],
                    'times': np.asarray(data['times']),
                    'num_waypoints': data['num_waypoints'],
                    'duration': data['duration'],
                    'forces': np.asarray(data['forces']) if 'forces' in data else None
                }
            else:
                raise ValueError(f"Unknown format: {format}")
            
            msg = f"[RECORDER] Loaded trajectory from {filepath} ({format} format)"
            msg += f"\n  - Waypoints: {trajectory['num_waypoints']}"
            msg += f"\n  - Duration: {trajectory['duration']:.2f} s"
            if trajectory['forces'] is not None:
                msg += f"\n  - With force data"
            print(msg)
            return trajectory
        
        except Exception as e:
            print(f"[RECORDER] Failed to load trajectory: {str(e)}")
            return None
