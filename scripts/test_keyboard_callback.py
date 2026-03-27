"""
Quick test to verify keyboard callback is working
"""
import numpy as np
import mujoco

# Simple test model
xml = """
<mujoco>
  <worldbody>
    <body pos="0 0 0">
      <inertial pos="0 0 0" mass="1" diaginv="1 1 1"/>
      <geom type="sphere" size="0.1" rgba="1 0 0 0.5"/>
    </body>
  </worldbody>
</mujoco>
"""

def test_key_callback():
    """Test that key callbacks work"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    
    # Track key events
    key_events = []
    
    def key_callback(keycode):
        key_events.append(keycode)
        print(f"Key event: {keycode}")
    
    print("Opening viewer with key callback...")
    print("Try pressing arrow keys!")
    print("Press Escape to close...\n")
    
    try:
        with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
            while viewer.is_running():
                mujoco.mj_step(m, d)
                viewer.sync()
    except KeyboardInterrupt:
        pass
    
    print(f"\nTotal key events captured: {len(key_events)}")
    print(f"Key codes: {key_events}")
    
    # Check for arrow keys (262-265)
    arrow_keys = [k for k in key_events if 262 <= abs(k) <= 265]
    print(f"Arrow key events: {arrow_keys}")
    
    if arrow_keys:
        print("✓ Key callback working correctly!")
    else:
        print("✗ No arrow key events captured - make sure to click on the viewer window")


if __name__ == "__main__":
    test_key_callback()
