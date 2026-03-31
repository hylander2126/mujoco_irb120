# Force Recording and Playback Guide

## Problem
When recording robot motion via keyboard control, the applied downward force/pressure was not being captured. During playback, the robot would follow the trajectory but **without** the force component.

## Solution A: Enhanced Force Recording and Playback

### What Changed

**Updated `TrajectoryRecorder` class** to support optional force/wrench recording:
- Added `record_forces` parameter to `start_recording()`
- Records F/T sensor readings (6D wrench: Fx, Fy, Fz, Tx, Ty, Tz) at each waypoint
- During playback, force data is interpolated and applied alongside position control
- Force data is saved/loaded with the trajectory file

### Usage in Notebook

```python
# Record with force tracking
MOTION_MODE = 'RECORD'
RECORD_FORCES = True    # Enable force recording

# Run the simulation with manual keyboard control
# Robot will record both joint positions AND force commands

# Playback will then apply both position and force
MOTION_MODE = 'PLAYBACK'
# Forces are automatically applied from the recorded trajectory
```

### How It Works

**Recording Phase:**
1. User applies keyboard control + manual downward pressure during simulation
2. TrajectoryRecorder captures:
   - Joint angles (as before)
   - F/T sensor reading at each step (new feature)
   - Timestamps

**Playback Phase:**
1. Trajectory loaded from disk with force data
2. Robot follows interpolated joint trajectory
3. F/T sensor readings are interpolated and replayed in parallel with position control
4. Result: Robot applies the same force profile as during recording

### Technical Details

- **Force Capture Method**: Uses `robot.ft_get_reading()` which returns bias-corrected wrench
- **Interpolation**: Linear interpolation between recorded force waypoints
- **File Format**: Forces stored in `.npz` file alongside trajectory waypoints
- **Backward Compatible**: If `record_forces=False`, only positions are recorded (original behavior)

### Limitations

- Force playback is **experimental** - depends on your robot controller's `ft_apply_wrench()` method
- Forces are replayed based on time interpolation, not closed-loop force control
- May not capture dynamic effects (tool compliance, sensor noise, etc.)

## Solution B: Workaround - Save All Simulation Variables

### What Changed

Added a new cell that saves **all** collected simulation data to a numpy file:

```python
np.savez(
    "simulation_data.npz",
    t_hist=t_hist,                  # Time array
    w_hist=w_hist,                  # Force/torque history (N, Nm)
    quat_hist=quat_hist,            # Object quaternion 
    ball_pose_hist=ball_pose_hist,  # Ball-center TCP frames
    sens_pose_hist=sens_pose_hist,  # F/T sensor frames
    con_bool_hist=con_bool_hist,    # Contact flags
    obj_pose_hist=obj_pose_hist,    # Object pose frames
    ball_pos_hist=ball_pos_hist,    # Ball position trajectory
    sens_pos_hist=sens_pos_hist,    # Sensor position trajectory
    obj_pos_hist=obj_pos_hist       # Object position trajectory
)
```

### When to Use

- If you just want to preserve all measurements from your experiment
- For post-processing analysis or visualization
- As a backup of complete experimental run data
- Loading for offline force profiling or control redesign

### Loading Saved Data

```python
data = np.load("simulation_data.npz")
w_history = data['w_hist']           # Force/torque at each step
time = data['t_hist']
obj_poses = data['obj_pose_hist']
# ... etc
```

---

## Quick Start

### Recording with Force

```python
# In main.ipynb
MOTION_MODE = 'RECORD'
RECORD_FORCES = True   # NEW: Enable force capture

# Run simulation with manual keyboard control
# Apply downward pressure as needed
```

### Replaying with Force

```python
MOTION_MODE = 'PLAYBACK'
# Recorded motion is replayed with both position AND force commands
```

### Accessing All Experimental Data

After the simulation runs, `simulation_data.npz` is automatically saved with all captured variables.

---

## Files Modified

1. **`src/mujoco_irb120/common/trajectory_recorder.py`**
   - Added `record_forces` parameter
   - Added `forces` list to store wrench data
   - Updated save/load methods to handle force data
   - Updated playback to apply forces

2. **`scripts/main.ipynb`**
   - Added `RECORD_FORCES` flag (line 80)
   - Updated recorder initialization to pass `record_forces=RECORD_FORCES`
   - Added new cell to save `simulation_data.npz` after each run

---

## Troubleshooting

**Q: Forces aren't being applied during playback**
- Check that `RECORD_FORCES=True` was set during recording
- Verify `recorded_motion.npz` contains force data: `data = np.load('recorded_motion.npz'); print(data.files)` should include 'forces'
- Force application requires `ft_apply_wrench()` support in your robot controller

**Q: Trajectory playback works but only position, no force**
- Trajectories recorded with `RECORD_FORCES=False` don't have force data
- Re-record with `RECORD_FORCES=True` to capture forces

**Q: Large file sizes**
- Force data increases file size significantly (~6x for wrench vs position-only)
- Use `simulation_data.npz` for analysis, `recorded_motion.npz` for playback

