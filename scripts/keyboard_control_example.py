"""
Example: Keyboard Cartesian Control for Robot ARM

This demonstrates how to use keyboard control to interactively move the robot.

Keyboard Controls:
- LEFT arrow:  Move X negative (away from object)
- RIGHT arrow: Move X positive (toward object)  
- DOWN arrow:  Move Z negative (lower end-effector)
- UP arrow:    Move Z positive (raise end-effector)

How it works:
1. The viewer's key_callback function captures keyboard events
2. Keyboard state is stored in a dictionary
3. get_keyboard_input() polls this state and returns velocity commands
4. apply_cartesian_keyboard_ctrl() integrates velocity into cartesian position

Running the example:
```python
python keyboard_control_example.py
```
Press arrow keys in the viewer window to control the robot!
"""

import numpy as np
import mujoco
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mujoco_irb120.common.load_obj_in_env import load_environment
from mujoco_irb120.common.robot_controller import controller
from mujoco_irb120.common.render_opts import RendererViewerOpts


def demo_keyboard_control():
    """Interactive demonstration of keyboard cartesian control"""
    
    print("=" * 60)
    print("Keyboard Cartesian Control Demo")
    print("=" * 60)
    print("\nControls:")
    print("  LEFT arrow ←   : Move -X (away from object)")
    print("  RIGHT arrow →  : Move +X (toward object)")
    print("  DOWN arrow ↓   : Move -Z (lower)")
    print("  UP arrow ↑     : Move +Z (lift)")
    print("\nClick on the viewer window to ensure it has focus!")
    print("=" * 60 + "\n")
    
    # Setup
    OBJECT = 14  # flashlight
    model, data = load_environment(num=OBJECT, launch_viewer=False)
    irb = controller(model, data)
    
    # Set initial pose
    T_home = irb.FK()
    T_init = T_home.copy()
    T_init[:3, 3] = np.array([0.374, 0, 0.225])
    
    print("Computing initial IK...")
    q_init = irb.IK(T_init, method=2, damping=0.5, max_iters=1000)
    irb.set_pose(q=q_init)
    print("Initial pose set.\n")
    
    # Interactive control parameters
    dt = model.opt.timestep  # MuJoCo timestep
    run_duration = 120.0  # 2 minutes for exploration
    
    print(f"Starting interactive control loop (max {run_duration}s)...\n")
    
    with RendererViewerOpts(model, data, vis=True) as rv:
        iteration = 0
        while rv.viewer_is_running() and data.time < run_duration:
            # Get keyboard input
            v_cmd = rv.get_keyboard_input()
            
            # Display status every 60 iterations (about once per second @ 500Hz)
            if iteration % 60 == 0 and v_cmd is not None and not np.allclose(v_cmd, 0):
                print(f"t={data.time:.2f}s  v_cmd: x={v_cmd[3]:+.2f} m/s, z={v_cmd[5]:+.2f} m/s")
            
            # Apply keyboard control
            if v_cmd is not None:
                success = irb.apply_cartesian_keyboard_ctrl(v_cmd, dt=dt, verbose=True)
                if not success and not np.allclose(v_cmd, 0):
                    print(f"  ⚠ IK failed - outside workspace")
            
            # Advance simulation
            mujoco.mj_step(model, data)
            rv.sync()
            iteration += 1
    
    print(f"\n Simulation ended at t={data.time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    demo_keyboard_control()

