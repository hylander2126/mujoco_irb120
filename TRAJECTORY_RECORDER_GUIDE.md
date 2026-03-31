# Trajectory Recorder Guide

## Overview

The trajectory recorder is a powerful feature that allows you to:
- **Record** keyboard inputs or automated trajectories as a series of waypoints
- **Playback** recorded trajectories with optional interpolation
- **Save/Load** trajectories to/from disk for repeated use
- **Waypoint-based** control with flexible recording options

## Quick Start

### Recording a Trajectory

1. **Set up configuration flags** in `main.ipynb`:
   ```python
   RECORD_TRAJECTORY = True       # Enable recording
   PLAYBACK_TRAJECTORY = False    # Disable playback
   KEYBOARD_CONTROL = True        # Enable keyboard control for input
   RECORD_TYPE = 'joints'         # Record joint positions
   TRAJECTORY_FILE = "my_trajectory.npz"
   RECORD_EVERY_N_STEPS = 1       # Record every simulation step
   ```

2. **Run the main simulation loop** - the robot will record waypoints as you control it with arrow keys

3. **Trajectory is automatically saved** to `TRAJECTORY_FILE` at the end (if file is specified)

### Playing Back a Trajectory

1. **Set up configuration flags**:
   ```python
   RECORD_TRAJECTORY = False      # Disable recording
   PLAYBACK_TRAJECTORY = True     # Enable playback
   PLAYBACK_TYPE = 'joints'       # Use joint position control
   PLAYBACK_INTERPOLATE = True    # Smooth waypoint interpolation
   TRAJECTORY_FILE = "my_trajectory.npz"  # File to load
   ```

2. **Run the main simulation loop** - the robot will replay the saved trajectory

## Configuration Flags

### Recording Flags
- **`RECORD_TRAJECTORY`** (bool): Enable/disable recording
- **`RECORD_TYPE`** (str): Type of data to record
  - `'joints'`: Joint configurations (6D) - smallest file size, most repeatable
  - `'pose'`: End-effector 4x4 transformation matrices - most precise
  - `'cartesian'`: End-effector position only (3D) - lightweight compromise
- **`RECORD_EVERY_N_STEPS`** (int): Record every N simulation steps
  - 1: Record every step (most waypoints, largest file)
  - 5: Record every 5th step (smaller file, less smooth playback)
  - 10: Record every 10th step (most compact, coarser playback)

### Playback Flags
- **`PLAYBACK_TRAJECTORY`** (bool): Enable/disable playback
- **`PLAYBACK_TYPE`** (str): Playback control method
  - `'joints'`: Direct joint position control (precise, uses recorded configs)
  - `'cartesian'`: Cartesian control with IK (flexible, can adapt poses)
- **`PLAYBACK_INTERPOLATE`** (bool): Interpolate between waypoints
  - `True`: Smooth linear interpolation between waypoints (recommended)
  - `False`: Jump to nearest waypoint (jerky, not recommended)
- **`PLAYBACK_MAINTAIN_ORI`** (bool): Hold orientation during cartesian playback

### File Handling
- **`TRAJECTORY_FILE`** (str): Path to save/load trajectory
  - Format: `.npz` for numpy (binary, compact)
  - Can also use `.json` format (text, human-readable)

## Recorder API Methods

All methods are available on the robot controller instance (`irb`).

### Recording Methods

#### `start_recording(verbose=True)`
Begin recording waypoints. Clears any previous recording.
```python
irb.start_recording(verbose=True)
```

#### `stop_recording(verbose=True)`
Stop recording and return summary statistics.
```python
summary = irb.stop_recording(verbose=True)
# Returns: {'num_waypoints': int, 'duration': float, 'waypoints': list, 'times': list}
```

#### `record_waypoint(record_type='joints')`
Manually record a single waypoint at current time.
```python
# During recording
irb.record_waypoint(record_type='joints')  # Record junction config
irb.record_waypoint(record_type='pose')    # Record end-effector pose
irb.record_waypoint(record_type='cartesian')  # Record EE position only
```

#### `get_recorded_trajectory()`
Get the currently recorded trajectory data.
```python
traj = irb.get_recorded_trajectory()
# Returns: {'waypoints': list, 'times': np.ndarray, 'num_waypoints': int, 'duration': float}
```

#### `clear_trajectory(verbose=True)`
Clear all recorded waypoints.
```python
irb.clear_trajectory(verbose=True)
```

### Playback Methods

#### `start_playback(trajectory=None, verbose=True)`
Begin playback of a trajectory.
```python
# Playback current recording
irb.start_playback(verbose=True)

# Playback a specific trajectory
irb.start_playback(trajectory=loaded_traj, verbose=True)
```

#### `stop_playback(verbose=True)`
Stop playback.
```python
irb.stop_playback(verbose=True)
```

#### `playback_step(playback_type='joints', interpolate=True, maintain_orientation=False)`
Execute one step of playback. Call every simulation step during playback.
```python
# In main loop
still_playing = irb.playback_step(
    playback_type='joints',
    interpolate=True,
    maintain_orientation=False
)
if not still_playing:
    print("Playback finished!")
```

### File I/O Methods

#### `save_trajectory(filepath, format='numpy')`
Save the current recorded trajectory to disk.
```python
irb.save_trajectory("trajectory_push.npz", format='numpy')
irb.save_trajectory("trajectory_push.json", format='json')
```

#### `load_trajectory(filepath, format='numpy')`
Load a previously saved trajectory.
```python
loaded_traj = irb.load_trajectory("trajectory_push.npz", format='numpy')
loaded_traj = irb.load_trajectory("trajectory_push.json", format='json')
```

## Usage Examples

### Example 1: Record and Immediately Playback

```python
# Configuration
RECORD_TRAJECTORY = True
KEYBOARD_CONTROL = True
RECORD_TYPE = 'joints'
RECORD_EVERY_N_STEPS = 1

# Run main loop - control with arrow keys to record trajectory
# At end of simulation:
irb.save_trajectory("temp_traj.npz")

# Then set for playback:
RECORD_TRAJECTORY = False
PLAYBACK_TRAJECTORY = True
TRAJECTORY_FILE = "temp_traj.npz"

# Run main loop again - robot replays your movements
```

### Example 2: Record Coarse Trajectory, Playback Smoothly

```python
# Record with lower frequency
RECORD_TRAJECTORY = True
RECORD_EVERY_N_STEPS = 5  # 5x fewer waypoints
KEYBOARD_CONTROL = True
TRAJECTORY_FILE = "coarse_trajectory.npz"

# Run recording loop...

# Playback with smooth interpolation
RECORD_TRAJECTORY = False
PLAYBACK_TRAJECTORY = True
PLAYBACK_INTERPOLATE = True  # Interpolate between sparse waypoints
TRAJECTORY_FILE = "coarse_trajectory.npz"

# Run playback loop - smooth despite coarse recording
```

### Example 3: Record Poses, Playback with IK Adaptation

```python
# Record end-effector poses
RECORD_TRAJECTORY = True
RECORD_TYPE = 'pose'  # Record 4x4 transforms
TRAJECTORY_FILE = "ee_trajectory.npz"

# Run recording loop with keyboard control...

# Playback with IK to handle slight variations
RECORD_TRAJECTORY = False
PLAYBACK_TRAJECTORY = True
PLAYBACK_TYPE = 'cartesian'  # Use IK for playback
TRAJECTORY_FILE = "ee_trajectory.npz"

# Run playback - robot will use IK to reach recorded poses
```

### Example 4: Manual Recording Session

```python
# Create and record a custom trajectory
model, data = load_environment(num=0, launch_viewer=False)
irb = robot_controller.controller(model, data)

irb.start_recording(verbose=True)

# Manually move robot and record waypoints
q1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
irb.set_pose(q=q1)
irb.record_waypoint(record_type='joints')

# ... move robot again ...
q2 = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
irb.set_pose(q=q2)
irb.record_waypoint(record_type='joints')

# Get summary
summary = irb.stop_recording(verbose=True)

# Save for later use
irb.save_trajectory("custom_trajectory.npz")
```

## Tips and Best Practices

1. **Choose Recording Type Wisely**
   - Use `'joints'` for repeatable trajectories within same object/scene
   - Use `'pose'` for context-independent trajectories
   - Use `'cartesian'` for lightweight storage

2. **Recording Frequency Tradeoff**
   - `RECORD_EVERY_N_STEPS = 1`: Smooth playback, larger files
   - `RECORD_EVERY_N_STEPS = 5-10`: Good compromise, manageable files
   - Higher values: Coarser playback, but enable with `PLAYBACK_INTERPOLATE=True`

3. **Playback Quality**
   - Always use `PLAYBACK_INTERPOLATE = True` for smoother playback
   - Use `PLAYBACK_TYPE = 'joints'` for accurate replay of recorded configs
   - Use `PLAYBACK_TYPE = 'cartesian'` if trajectory needs adaptation

4. **File Management**
   - Use `.npz` format for binary efficiency (5-10x smaller than `.json`)
   - Use `.json` format if you need human-readable trajectory data
   - Include trajectory name/date in filename: `trajectory_push_2024_03_30.npz`

5. **Debugging Recording Issues**
   - Check `irb.waypoints` to see recorded data
   - Check `irb.waypoint_times` to see timing
   - Use `irb.get_recorded_trajectory()` for full summary
   - Print first few waypoints to verify data

## Limitations and Notes

- Playback is currently deterministic (replays exact same trajectory)
- For real-world deployment, consider adding small perturbations/noise to recorded trajectories
- IK-based playback (`PLAYBACK_TYPE='cartesian'`) may fail if target poses are outside manipulability ellipsoid
- Time-based interpolation assumes linear variation between waypoints
- Recorded trajectories are robot-specific (same joint limits, DOF configuration required)

## Troubleshooting

**Issue: Playback not starting**
- Check `PLAYBACK_TRAJECTORY = True`
- Verify `TRAJECTORY_FILE` path is correct
- Ensure file exists if loading

**Issue: Jerky playback**
- Enable `PLAYBACK_INTERPOLATE = True`
- Reduce `RECORD_EVERY_N_STEPS` when recording
- Check robot joint velocity limits

**Issue: IK failures during cartesian playback**
- Reduce number of recorded poses (coarser recording)
- Check target poses are within manipulability ellipsoid
- Use `PLAYBACK_TYPE = 'joints'` instead

**Issue: Large file sizes**
- Increase `RECORD_EVERY_N_STEPS` (record less frequently)
- Use `RECORD_TYPE = 'cartesian'` instead of `'pose'`
- Use `.npz` format instead of `.json`
