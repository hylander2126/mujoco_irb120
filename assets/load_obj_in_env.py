import mujoco
import mujoco.viewer
import os

# -----------------------------------------------------------------
# 1. DEFINE YOUR OBJECT CONFIGURATIONS
# -----------------------------------------------------------------
# Based on your file structure and table position.
#
# Your table's top surface is at z=0.05 (body_z=0 + geom_half_height=0.05).
# I'll set the object's 'pos' z-value to 0.06, so it spawns
# 1cm above the table, centered at (0.75, 0.0).

OBJECT_CONFIGS = {
    1: {
        "name": "wristwatch",
        "path": "lib/object_sim/wristwatch/body.xml",
        "assets": "lib/object_sim/wristwatch/assets.xml",
        "pos": "0.75 0.0 0.06",
        "quat": "1 0 0 0",
    },
    2: {
        "name": "toothbrush",
        "path": "lib/object_sim/toothbrush/body.xml",
        "assets": "lib/object_sim/toothbrush/assets.xml",
        "pos": "0.75 0.1 0.06",
        "quat": "0.707 0.707 0 0",
    },
    3: {
        "name": "waterbottle",
        "path": "lib/object_sim/waterbottle/body.xml",
        "assets": "lib/object_sim/waterbottle/assets.xml",
        "pos": "0.70 -0.1 0.06",
        "quat": "1 0 0 0",
    },

    # ... add your other 7 objects here
}

# -----------------------------------------------------------------
# 2. CREATE THE XML-GENERATING FUNCTION
# -----------------------------------------------------------------
def create_scene_xml(object_id, template_path="table_push.xml"):
    """
    Generates a complete scene XML by combining the template
    with a specific object's configuration.
    """
    if object_id not in OBJECT_CONFIGS:
        raise ValueError(f"Object ID {object_id} not defined in OBJECT_CONFIGS")
    cfg = OBJECT_CONFIGS[object_id]

    # Create the XML block string for the object.
    # This body will wrap the included object, setting its pose.
    object_xml_block = f"""
    <body name="{cfg['name']}_base" pos="{cfg['pos']}" quat="{cfg['quat']}">
        <include file="{cfg['path']}"/>
    </body>
    """


    # This pulls in the object's meshes/textures/materials at top-level
    asset_xml_block = f'<include file="{cfg["assets"]}"/>'


    # Load the template file
    try:
        with open(template_path, "r") as f:
            template = f.read()
    except FileNotFoundError:
        print(f"Error: Template file not found at '{template_path}'")
        print("Make sure 'run_experiment.py' is in the same directory as 'table_push.xml'")
        return None

    # Use Python's .format() to inject the object block
    final_xml_content = template.format(
        object_block=object_xml_block,
        asset_block=asset_xml_block
    )

    # Save the final, combined XML to a new file
    out = "generated_scene.xml"
    with open(out, "w") as f:
        f.write(final_xml_content)

    print(f"Generated '{out}' with {cfg['name']} at {cfg['pos']}")
    return out

# -----------------------------------------------------------------
# 3. RUN YOUR EXPERIMENT
# -----------------------------------------------------------------

def load(num=1, launch_viewer=False):
    xml_path = create_scene_xml(num)
    if xml_path:
        try:
            m = mujoco.MjModel.from_xml_path(xml_path)
            d = mujoco.MjData(m)

            if launch_viewer:
                with mujoco.viewer.launch_passive(m, d) as viewer:
                    while viewer.is_running():
                        mujoco.mj_step(m, d)
                        viewer.sync()
            return m, d
        except Exception as e:
            print(f"Error loading or running simulation: {e}")
    return None, None


# allow running directly
if __name__ == "__main__":
    load(num=2, launch_viewer=True)