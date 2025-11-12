import mujoco
import mujoco.viewer
import os
from pathlib import Path
from xml.etree import ElementTree as ET

# -----------------------------------------------------------------
# 1. DEFINE YOUR OBJECT CONFIGURATIONS
# -----------------------------------------------------------------
#
# Table top surface is at 0.1m in z, but each object has its own frame, 
# so this has to be done empirically, for now

OBJECT_CONFIGS = {
    0: {"name": "box_exp"},
    1: {"name": "alarmclock",  "pos": "1.0 0.00 0.225", "quat": "1 0 0 0",  'scale': "1.0"},
    2: {"name": "binoculars",  "pos": "1.0 0.00 0.225", "quat": "1 0 0 0",  'scale': "1.0"},
    3: {"name": "camera",      "pos": "0.75 0.00 0.15", "quat": "1 0 0 0",  'scale': "1.0"},
    4: {"name": "elephant",    "pos": "1.0 0.00 0.225", "quat": "1 0 0 0",  'scale': "1.0"},
    5: {"name": "flashlight",  "pos": "1.0 0.00 0.225", "quat": "1 0 0 0",  'scale': "2.0"},
    6: {"name": "hammer",      "pos": "1.0 0.00 0.225", "quat": "1 0 0 0",  'scale': "1.0"},
    7: {"name": "waterbottle", "pos": "1.00 0.00 0.15", "quat": "1 0 0 0",  'scale': "2.0"},
    8: {"name": "wineglass",   "pos": "0.75 0.00 0.15", "quat": "1 0 0 0",  'scale': "1.0"},
    # ... add your other 7 objects here
}


REPO_ROOT = Path(__file__).resolve().parents[1] # Adjust if needed
ASSETS_DIR = REPO_ROOT / "assets"
OBJ_DIR = ASSETS_DIR / "object_sim"
GEN_DIR = ASSETS_DIR / "_generated"

# -----------------------------------------------------------------
# 2. CREATE THE XML-GENERATING FUNCTION
# -----------------------------------------------------------------
def create_scene_xml(
        object_id, 
        template_path = str(ASSETS_DIR / "table_push.xml"),
        out           = str(ASSETS_DIR / "generated_scene.xml")
    ):
    
    if not object_id:
        asset_block = ""
        object_block = '<include file="my_objects/box/box_exp.xml"/>'
    else:
        print(f"Current directory: {os.getcwd()}")
        cfg = OBJECT_CONFIGS[object_id]
        name = cfg["name"]
        asset_path = OBJ_DIR / name / "assets.xml"
        body_path  = OBJ_DIR / name / "body.xml"
        scaled_path = GEN_DIR / name / "assets_scaled.xml"

        # Perform scaling per-object by generating a scaled copy of assets.xml
        write_scaled_assets_copy(asset_path, scaled_path, cfg["scale"])
        asset_include = f'<include file="{scaled_path.as_posix()}"/>'

        ## IMPORTANT: use absolute meshdir and absolute includes (use our common_modified.xml)
        asset_block = f"""
        <compiler meshdir="{OBJ_DIR.as_posix()}"/>
        <include file="{(ASSETS_DIR / "common_modified.xml").as_posix()}"/>
        {asset_include}
        """

        # Make sure to set the childclass to "grab" and set the joint to "free" so it's not 'welded'
        object_block = f"""
        <body name="{name}_base" pos="{cfg['pos']}" quat="{cfg['quat']}" childclass="grab">
            <joint type="free"/>
            <site name="payload_site" pos="0 0 0" size="0.02 0.02 0.02" type="box" rgba="1 1 0 0"></site>
            <site name="obj_frame_site" pos="0.05 0.0 -0.15" size="0.02 0.02 0.02" type="box" rgba="1 0 0 0"></site>
            <include file="{body_path}"/>
        </body>
        """

    with open(template_path, "r") as f:
        tpl = f.read()
    with open(out, "w") as f:
        f.write(tpl.format(asset_block=asset_block, object_block=object_block))
    return out


def write_scaled_assets_copy(vendor_asset_path: Path, out_path: Path, scale: float) -> Path:
    """
    Copy vendor assets.xml to out_path, applying scale to every <mesh>.
    Creates parent dirs as needed. Returns out_path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse (works with <mujocoinclude> or plain <asset>)
    tree = ET.parse(vendor_asset_path)
    root = tree.getroot()

    # Find all mesh tags anywhere under root
    for mesh in root.findall(".//mesh"):
        mesh.set("scale", f"{scale} {scale} {scale}")

    # Write back
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    return out_path


# -----------------------------------------------------------------
# 3. RUN EXPERIMENT
# -----------------------------------------------------------------

def load_environment(num=1, launch_viewer=False):
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