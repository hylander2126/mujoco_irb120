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
    0: {"name": "box",          "pos": "0.1 0.0 0.2",   "euler": "0 0 0",   "rgba": "1 0 0 1"},
    1: {"name": "alarmclock",   "pos": "1.0 0.0 0.1",   "quat": "1 0 0 0",  'scale': "1.0"},
    2: {"name": "binoculars",   "pos": "1.0 0.0 0.1",   "quat": "1 0 0 0",  'scale': "1.0"},
    3: {"name": "camera",       "pos": "1.0 0.0 0.1",   "quat": "1 0 0 0",  'scale': "1.0"},
    4: {"name": "elephant",     "pos": "1.0 0.0 0.1",   "quat": "1 0 0 0",  'scale': "1.0"},
    # 5: {"name": "flashlight",   "pos": "1.0 0.0 0.225", "quat": "1 0 0 0",  'scale': "2.0", "euler": "0 0 0", "rgba": "0.8 0.8 0.2 1"},
    6: {"name": "hammer",       "pos": "1.0 0.0 0.225", "quat": "0.707107 0 -0.707107 0",  'scale': "1.0"},
    7: {"name": "waterbottle",  "pos": "1.0 0.0 0.2",   "quat": "1 0 0 0",  'scale': "2.0"},
    8: {"name": "wineglass",    "pos": "1.0 0.0 0.14",  "quat": "1 0 0 0",  'scale': "1.0"},
    10: {"name": "heart",       "pos": "0.2 0.0 0.05",  "euler": "0 0 0",   "rgba": "1 0 0 1"},
    11: {"name": "L",           "pos": "0.45 0.0 0.05", "euler": "0 0 0",   "rgba": "1 0 0 1"},
    12: {"name": "monitor",     "pos": "0.5 0.0 0.05",  "euler": "0 0 0",   "rgba": "0.1 0.1 0.1 1"},
    13: {"name": "soda",        "pos": "0.75 0.0 0.05", "euler": "1.5719 0 0", "rgba": "0 0.6 0.6 0.6"},
    14: {"name": "flashlight",  "pos": "1.0 0.0 0.225", "euler": "0 0 0",   "rgba": "0.9 0.1 0.1 1", 'scale': "1.0"},
}

ACTUATOR_BLOCK = f"""
<actuator>
    <!-- Position Control -->
    <!-- kp, kv: (200,100) first 3, (100,50) last 3 -->
    <position joint="joint_1" name="joint_1" kp="200" kv="100" ctrlrange="-2.87979 2.87979" forcerange="-20 20"/>
    <position joint="joint_2" name="joint_2" kp="200" kv="100" ctrlrange="-1.91986 1.91986" forcerange="-20 20"/>
    <position joint="joint_3" name="joint_3" kp="200" kv="100" ctrlrange="-1.22173 1.91986" forcerange="-20 20"/>
    <position joint="joint_4" name="joint_4" kp="100" kv="50" ctrlrange="-2.79252 2.79252" forcerange="-10 10"/>
    <position joint="joint_5" name="joint_5" kp="100" kv="50" ctrlrange="-2.09440 2.90440" forcerange="-10 10"/>
    <position joint="joint_6" name="joint_6" kp="100" kv="50" ctrlrange="-3.14200 3.14200" forcerange="-10 10"/>
</actuator>

<sensor>
    <force name="force_sensor" site="sensor_site"/>
</sensor>
"""


REPO_ROOT = Path(__file__).resolve().parents[1]
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
    
    if object_id in [0, 10, 11, 12, 13, 14
    ]:
        name = OBJECT_CONFIGS[object_id]["name"]
        asset_block = f'<include file="{(ASSETS_DIR / "common_modified.xml").as_posix()}"/>'
        object_block = f"""
        <include file="my_objects/robot/robot.xml"> </include>

        <include file="my_objects/{name}/{name}_exp.xml"/>
        """

    else:
        print(f"Current directory: {os.getcwd()}")
        cfg = OBJECT_CONFIGS[object_id]
        name = cfg["name"]
        asset_path = OBJ_DIR / name / "assets.xml"
        body_path  = OBJ_DIR / name / "body.xml"
        scaled_path = GEN_DIR / name / "assets_scaled.xml"

        # Perform scaling per-object by generating a scaled copy of assets.xml
        write_scaled_assets_copy(asset_path, scaled_path, cfg["scale"])
        scaled_asset_include = f'<include file="{scaled_path.as_posix()}"/>'

        ## IMPORTANT: use absolute meshdir and absolute includes (use our common_modified.xml)
        asset_block = f"""
        <compiler meshdir="{OBJ_DIR.as_posix()}"/>
        <include file="{(ASSETS_DIR / "common_modified.xml").as_posix()}"/>
        {scaled_asset_include}
        """

        # Make sure to set the childclass to "grab" and set the joint to "free" so it's not 'welded'
        object_block = f"""
        <include file="my_objects/robot/robot.xml"> </include>

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
        f.write(tpl.format(actuator_block=ACTUATOR_BLOCK, asset_block=asset_block, object_block=object_block))
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








# -----------------------------------------------------------------
# 4. PHOTOSHOOT UTILS
# -----------------------------------------------------------------

def create_photoshoot_xml(
        nums            =[0, 10, 11, 12, 13, 5],
        template_path   = str(ASSETS_DIR / "table_push.xml"),
        out             = str(ASSETS_DIR / "photoshoot_scene.xml")
    ):
    # asset_path = OBJ_DIR / name / "assets.xml"
    asset_block = f"""
    <compiler meshdir="{OBJ_DIR.as_posix()}"/>
    <include file="{(ASSETS_DIR / "common_modified.xml").as_posix()}"/>
    <include file="{(OBJ_DIR / "flashlight" / "assets.xml").as_posix()}"/>
    """

    # Build object block: one body per object
    object_blocks = []
    for i, oid in enumerate(nums):
        cfg = OBJECT_CONFIGS[oid]
        name = cfg["name"]
        pos = cfg["pos"]
        rpy = cfg["euler"]
        rgba = cfg["rgba"]

        if oid == 0:
            block = f"""
            <body name="{name}_base" pos="{pos}" euler="{rpy}">
                <geom name="payload" type="box" mass="0.615" size="0.05 0.05 0.15" material="block_mat"/>
            </body>
            """
        elif oid == 5: # For the flashlight, use their custom process...
            body_path  = OBJ_DIR / name / "body.xml"
            block = f"""
            <body name="{name}_base" pos="0.7 0.0 0.12" quat="{cfg['quat']}" childclass="grab">
                <geom name="flashlight_visual" mesh="flashlight" rgba="{rgba}" />
            </body>
            """
        else:
            block = f"""
            <body name="{name}_base" pos="{pos}" euler="{rpy}">
                <geom name="{name}" type="mesh" mesh="{name}_exp" rgba="{rgba}"/>
            </body>
            """
        object_blocks.append(block)
        continue

    object_block = "\n".join(object_blocks)

    with open(template_path, "r") as f:
        tpl = f.read()
    with open(out, "w") as f:
        f.write(tpl.format(actuator_block='', asset_block=asset_block, object_block=object_block))
    return out



def load_photoshoot(nums=[0, 10, 11, 12, 13, 5], launch_viewer=True):
    """
    Create + load + optionally view the photoshoot scene.
    """
    xml_path = create_photoshoot_xml(nums)

    try:
        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)

        if launch_viewer:
            with mujoco.viewer.launch_passive(m, d) as viewer:
                print("[photoshoot] Scene loaded. Adjust camera + screenshot.")
                while viewer.is_running():
                    mujoco.mj_step(m, d)
                    viewer.sync()
        return m, d
    except Exception as e:
        print(f"[photoshoot] Error: {e}")
        return None, None
