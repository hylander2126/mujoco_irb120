import mujoco
import mujoco.viewer
import os

# Adjust path if needed
MODEL_PATH = "assets/table_push.xml"

def main():
    # Load the model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    # Launch interactive viewer with UI disabled
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True) as viewer:
        print("[INFO] Viewer launched without sidebars. Close the window to exit.")
        
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()