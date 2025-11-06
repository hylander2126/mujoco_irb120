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
    1: {"name": "wristwatch",  "pos": "0.75 0.00 0.06", "quat": "1 0 0 0"},
    2: {"name": "toothbrush",  "pos": "0.75 0.10 0.06", "quat": "0.707 0.707 0 0"},
    3: {"name": "waterbottle", "pos": "0.70 -0.10 0.06", "quat": "1 0 0 0"},
    # ... add your other 7 objects here
}

COMMON_DEFAULTS = "object_sim/common.xml"
SHARED_DEFAULTS = "shared.xml"

# -----------------------------------------------------------------
# 2. CREATE THE XML-GENERATING FUNCTION
# -----------------------------------------------------------------
def create_scene_xml(object_id, template_path="table_push.xml", out="generated_scene.xml"):
    """out also defines the ROOT of all relative paths in the generated XML."""
    cfg = OBJECT_CONFIGS[object_id]
    name = cfg["name"]

    asset_block = f"""
    <include file="assets/object_sim/common.xml"/>
    <compiler meshdir="assets/object_sim/{name}" texturedir="assets/object_sim/{name}"/>
    <include file="assets/object_sim/{name}/assets.xml"/>
    <compiler meshdir="assets/meshes" texturedir="assets/textures"/>
    """

    object_block = f"""
    <body name="{name}_base" pos="{cfg['pos']}" quat="{cfg['quat']}">
      <include file="assets/object_sim/{name}/body.xml"/>
    </body>
    """

    with open(template_path, "r") as f:
        tpl = f.read()
    with open(out, "w") as f:
        f.write(tpl.format(asset_block=asset_block, object_block=object_block))
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
    load(num=1, launch_viewer=True)