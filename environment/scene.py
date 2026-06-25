import mujoco
import mujoco.viewer
import copy
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
    0: {"name": "box",          "pos": "0.1 0.0 0.2",   "euler": "0 0 0",   "rgba": "1 0 0 0.9"},
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
    <force name="force_sensor" site="site:sensor"/>
    <torque name="torque_sensor" site="site:sensor"/>
</sensor>
"""


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ASSETS_DIR = REPO_ROOT / "robot" / "assets"
CUSTOM_OBJ_DIR = ROBOT_ASSETS_DIR / "objects"
OBJ_DIR = CUSTOM_OBJ_DIR / "object_sim"
GEN_DIR = ROBOT_ASSETS_DIR / "generated" / "scaled_objects"


CUSTOM_OBJECT_IDS = {0, 10, 11, 12, 13, 14}
ROBOT_MESH_NAMES = ("base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6", "pusher_link")
MATERIAL_RGBA = {
    "floor_mat": "0.16 0.18 0.20 1",
    "table_mat": "0.57 0.75 0.83 0.8",
    "block_mat": "1 0.2 0.2 1",
    "robot0:gripper_mat": "0.12 0.12 0.12 1",
}
OBJECT_VISUAL_RGBA = {
    "alarmclock": "0.22 0.25 0.25 1",
    "apple": "0.82 0.08 0.05 1",
    "banana": "1.0 0.85 0.08 1",
    "binoculars": "0.05 0.07 0.08 1",
    "bowl": "0.95 0.88 0.72 1",
    "camera": "0.04 0.04 0.05 1",
    "coffeemug": "0.92 0.92 0.88 1",
    "cup": "0.95 0.72 0.80 1",
    "duck": "1.0 0.78 0.05 1",
    "elephant": "0.55 0.58 0.60 1",
    "flashlight": "1.0 0.52 0.05 1",
    "gamecontroller": "0.08 0.08 0.10 1",
    "hammer": "0.45 0.32 0.18 1",
    "human": "0.76 0.58 0.46 1",
    "mouse": "0.08 0.08 0.09 1",
    "phone": "0.02 0.03 0.05 1",
    "rubberduck": "1.0 0.78 0.05 1",
    "stapler": "0.18 0.18 0.20 1",
    "table": "0.52 0.36 0.22 1",
    "teapot": "0.78 0.42 0.24 1",
    "toothbrush": "0.2 0.55 0.95 1",
    "waterbottle": "0.18 0.55 0.90 0.75",
    "wineglass": "0.85 0.95 1.0 0.45",
}
DEFAULT_OBJECT_RGBA = "0.88 0.48 0.18 1"
OBJECT_COLLISION_RGBA = "0.25 0.40 0.55 0.35"


def _rel_asset_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROBOT_ASSETS_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _append_all(parent, children):
    for child in children:
        parent.append(copy.deepcopy(child))


def _parse_children(path: Path):
    return list(ET.parse(path).getroot())


def _resolve_mesh_file(mesh_file: str, meshdir: Path) -> str:
    path = Path(mesh_file)
    if not path.is_absolute():
        path = meshdir / path
    return _rel_asset_path(path)


def _rgba_to_tuple(rgba: str):
    return tuple(float(v) for v in rgba.split())


def get_object_rgba(object_id):
    cfg = OBJECT_CONFIGS[object_id]
    if "rgba" in cfg:
        return _rgba_to_tuple(cfg["rgba"])
    return _rgba_to_tuple(OBJECT_VISUAL_RGBA.get(cfg["name"], DEFAULT_OBJECT_RGBA))


def _meshdir_for_output(out: Path) -> str:
    meshdir = os.path.relpath(ROBOT_ASSETS_DIR.resolve(), out.parent.resolve())
    return Path(meshdir).as_posix()


class GeneratedSceneBuilder:
    """
    Build a fully expanded MJCF scene for MuJoCo or Genesis.

    The output intentionally avoids <include> tags. Mesh file paths are written
    relative to robot/assets so the generated file is portable inside the repo.
    """

    def __init__(self, backend="mujoco"):
        if backend not in {"mujoco", "genesis"}:
            raise ValueError("backend must be 'mujoco' or 'genesis'")
        self.backend = backend

    def build(self, object_id=0):
        self._mesh_name_map = {}
        root = ET.Element("mujoco", {"model": f"irb120_{self.backend}_scene"})
        ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "."})

        self._add_options(root)
        self._add_actuators_and_sensors(root)
        ET.SubElement(root, "size", {"njmax": "500", "nconmax": "100"})

        asset = ET.SubElement(root, "asset")
        self._add_common_assets(asset)
        self._add_robot_assets(asset)
        self._add_object_assets(asset, object_id)

        self._add_defaults(root)

        worldbody = ET.SubElement(root, "worldbody")
        self._add_world(worldbody)
        self._add_robot_body(worldbody)
        self._add_object_body(worldbody, object_id)
        self._add_light(worldbody)
        self._apply_direct_geom_colors(root, object_id)
        self._sanitize_for_backend(root)

        ET.indent(root, space="    ")
        return ET.ElementTree(root)

    def build_object(self, object_id=0):
        self._mesh_name_map = {}
        object_name = OBJECT_CONFIGS[object_id]["name"]
        root = ET.Element("mujoco", {"model": f"{object_name}_{self.backend}_object"})
        ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "."})

        asset = ET.SubElement(root, "asset")
        self._add_common_assets(asset)
        self._add_object_assets(asset, object_id)
        self._add_defaults(root)

        worldbody = ET.SubElement(root, "worldbody")
        self._add_object_body(worldbody, object_id)
        self._apply_direct_geom_colors(root, object_id)
        self._sanitize_for_backend(root)

        ET.indent(root, space="    ")
        return ET.ElementTree(root)

    def build_robot(self):
        root = ET.Element("mujoco", {"model": f"irb120_{self.backend}_robot"})
        ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "."})

        asset = ET.SubElement(root, "asset")
        self._add_common_assets(asset)
        self._add_robot_assets(asset)

        worldbody = ET.SubElement(root, "worldbody")
        self._add_robot_body(worldbody)
        self._sanitize_for_backend(root)

        ET.indent(root, space="    ")
        return ET.ElementTree(root)

    def write(self, object_id=0, out=ROBOT_ASSETS_DIR / "generated_scene.xml"):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tree = self.build(object_id)
        tree.getroot().find("compiler").set("meshdir", _meshdir_for_output(out))
        tree.write(out, encoding="utf-8", xml_declaration=True)
        return str(out)

    def write_object(self, object_id=0, out=None):
        out = Path(out) if out is not None else ROBOT_ASSETS_DIR / "generated" / f"genesis_object_{object_id}.xml"
        out.parent.mkdir(parents=True, exist_ok=True)
        tree = self.build_object(object_id)
        tree.getroot().find("compiler").set("meshdir", _meshdir_for_output(out))
        tree.write(out, encoding="utf-8", xml_declaration=True)
        return str(out)

    def write_robot(self, out=None):
        out = Path(out) if out is not None else ROBOT_ASSETS_DIR / "generated" / "genesis_robot.xml"
        out.parent.mkdir(parents=True, exist_ok=True)
        tree = self.build_robot()
        tree.getroot().find("compiler").set("meshdir", _meshdir_for_output(out))
        tree.write(out, encoding="utf-8", xml_declaration=True)
        return str(out)

    def _add_options(self, root):
        option = ET.SubElement(
            root,
            "option",
            {
                "timestep": "0.001",
                "tolerance": "1e-9",
                "integrator": "implicitfast",
                "iterations": "100",
                "ls_iterations": "100",
            },
        )
        ET.SubElement(option, "flag", {"warmstart": "enable", "multiccd": "enable"})

        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "720"})

    def _add_actuators_and_sensors(self, root):
        root.extend(ET.fromstring(f"<root>{ACTUATOR_BLOCK}</root>"))

    def _add_common_assets(self, asset):
        ET.SubElement(asset, "material", {"name": "MatObject", "specular": "0.75", "shininess": "0.1", "reflectance": "0.5", "rgba": "0.9 0.7 0.5 1"})
        ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient", "rgb1": "0.45 0.6 0.807", "rgb2": "0.46 0.87 0.58", "width": "32", "height": "32"})
        ET.SubElement(asset, "texture", {"name": "grid", "type": "2d", "builtin": "checker", "rgb1": ".1 .2 .3", "rgb2": ".2 .3 .4", "width": "300", "height": "300"})
        ET.SubElement(asset, "material", {"name": "floor_mat", "specular": "0", "shininess": "0.5", "reflectance": "0", "rgba": "0.2 0.2 0.2 1"})
        ET.SubElement(asset, "material", {"name": "table_mat", "specular": "0", "shininess": "0.5", "reflectance": "0", "rgba": "0.57 0.75 0.83 0.8"})
        ET.SubElement(asset, "material", {"name": "block_mat", "specular": "0", "shininess": "0.5", "reflectance": "0", "rgba": "1 0.2 0.2 1"})
        ET.SubElement(asset, "material", {"name": "robot0:geomMat", "shininess": "0.03", "specular": "0.4"})
        ET.SubElement(asset, "material", {"name": "robot0:gripper_finger_mat", "shininess": "0.03", "specular": "0.4", "reflectance": "0"})
        ET.SubElement(asset, "material", {"name": "robot0:gripper_mat", "shininess": "0.03", "specular": "0.4", "reflectance": "0"})
        ET.SubElement(asset, "material", {"name": "robot0:base_mat", "shininess": "0.03", "specular": "0.4", "reflectance": "0"})
        ET.SubElement(asset, "material", {"name": "grid_mat", "texture": "grid", "texrepeat": "8 8", "reflectance": "0.2"})

    def _add_robot_assets(self, asset):
        robot_dir = ROBOT_ASSETS_DIR / "robot"
        for name in ROBOT_MESH_NAMES:
            visual_path = robot_dir / "visual" / f"{name}.stl"
            mesh_path = visual_path if visual_path.exists() else robot_dir / f"{name}.stl"
            ET.SubElement(asset, "mesh", {"name": name, "file": _rel_asset_path(mesh_path)})

    def _add_object_assets(self, asset, object_id):
        cfg = OBJECT_CONFIGS[object_id]
        name = cfg["name"]

        if object_id in CUSTOM_OBJECT_IDS:
            if name != "box":
                ET.SubElement(asset, "mesh", {"name": f"{name}_exp", "file": f"objects/{name}/{name}_exp.stl"})
            return

        mesh_name_map = {}
        asset_root = ET.parse(OBJ_DIR / name / "assets.xml").getroot()
        for mesh in asset_root.findall(".//mesh"):
            old_name = mesh.attrib["name"]
            new_name = f"{name}_{old_name}"
            mesh_name_map[old_name] = new_name
            attrs = dict(mesh.attrib)
            attrs["name"] = new_name
            attrs["file"] = _resolve_mesh_file(attrs["file"], OBJ_DIR)
            scale = str(cfg.get("scale", "1.0"))
            attrs["scale"] = f"{scale} {scale} {scale}"
            ET.SubElement(asset, "mesh", attrs)

        self._mesh_name_map = mesh_name_map

    def _add_defaults(self, root):
        default = ET.SubElement(root, "default")
        grab = ET.SubElement(default, "default", {"class": "grab"})
        free_joint_loss = "0" if self.backend == "genesis" else "0.001"
        ET.SubElement(grab, "joint", {"limited": "false", "margin": "0.01", "armature": free_joint_loss, "damping": "0", "frictionloss": free_joint_loss})
        ET.SubElement(grab, "geom", {"type": "mesh", "rgba": ".93 .99 .97 1.0"})
        ET.SubElement(grab, "site", {"size": "0.005 0 0", "rgba": "0.4 0.9 0.4 1"})
        object_col = ET.SubElement(grab, "default", {"class": "object_col"})
        ET.SubElement(object_col, "geom", {"type": "mesh", "density": "1250", "contype": "1", "conaffinity": "1", "friction": "1 0.5 0.01", "margin": "0.0005", "condim": "4", "rgba": ".3 .4 .5 1", "group": "3"})

    def _add_world(self, worldbody):
        ET.SubElement(worldbody, "geom", {"name": "floor", "size": "5 5 5", "type": "plane", "material": "grid_mat"})
        table = ET.SubElement(worldbody, "body", {"pos": "0.75 0 0", "name": "table0"})
        ET.SubElement(table, "geom", {"size": "1 0.5 0.05", "type": "box", "name": "table", "mass": "2000", "material": "table_mat", "friction": "0 0 0"})
        ET.SubElement(table, "site", {"name": "site:table", "pos": "0 0 0", "size": "0.02 0.02 0.02", "type": "box", "rgba": "1 1 0 0"})

    def _add_robot_body(self, worldbody):
        _append_all(worldbody, _parse_children(ROBOT_ASSETS_DIR / "robot" / "robot.xml"))

    def _add_object_body(self, worldbody, object_id):
        cfg = OBJECT_CONFIGS[object_id]
        name = cfg["name"]

        if object_id in CUSTOM_OBJECT_IDS:
            path = CUSTOM_OBJ_DIR / name / f"{name}_exp.xml"
            _append_all(worldbody, _parse_children(path))
            return

        body = ET.SubElement(
            worldbody,
            "body",
            {"name": f"{name}_base", "pos": cfg["pos"], "quat": cfg["quat"], "childclass": "grab"},
        )
        ET.SubElement(body, "joint", {"type": "free"})
        ET.SubElement(body, "site", {"name": "site:payload", "pos": "0 0 0", "size": "0.02 0.02 0.02", "type": "box", "rgba": "1 1 0 0"})
        ET.SubElement(body, "site", {"name": "site:obj_frame", "pos": "0.05 0.0 -0.15", "size": "0.02 0.02 0.02", "type": "box", "rgba": "1 0 0 0"})

        mesh_name_map = getattr(self, "_mesh_name_map", {})
        for child in _parse_children(OBJ_DIR / name / "body.xml"):
            cloned = copy.deepcopy(child)
            for geom in cloned.iter("geom"):
                mesh_name = geom.attrib.get("mesh")
                if mesh_name in mesh_name_map:
                    geom.set("mesh", mesh_name_map[mesh_name])
            body.append(cloned)

    def _add_light(self, worldbody):
        ET.SubElement(worldbody, "light", {"directional": "true", "ambient": "0.2 0.2 0.2", "diffuse": "0.8 0.8 0.8", "specular": "0.3 0.3 0.3", "castshadow": "false", "pos": "0 0 4", "dir": "0 0 -1", "name": "light0"})

    def _apply_direct_geom_colors(self, root, object_id):
        object_name = OBJECT_CONFIGS[object_id]["name"]
        object_rgba = OBJECT_VISUAL_RGBA.get(object_name, DEFAULT_OBJECT_RGBA)

        for geom in root.iter("geom"):
            material = geom.attrib.get("material")
            if material in MATERIAL_RGBA:
                geom.attrib.setdefault("rgba", MATERIAL_RGBA[material])

            if geom.attrib.get("class") == "object_col":
                geom.attrib.setdefault("rgba", OBJECT_COLLISION_RGBA)
                continue

            geom_name = geom.attrib.get("name", "")
            mesh_name = geom.attrib.get("mesh", "")
            if geom_name.endswith("_visual") or mesh_name.startswith(f"{object_name}_"):
                geom.attrib.setdefault("rgba", object_rgba)

            if geom_name in {"floor", "table"} and "rgba" not in geom.attrib:
                geom.set("rgba", MATERIAL_RGBA[f"{geom_name}_mat"])

    def _sanitize_for_backend(self, root):
        if self.backend != "genesis":
            return

        for joint in root.iter("joint"):
            if joint.attrib.get("type") == "free":
                joint.attrib.pop("armature", None)
                joint.attrib.pop("damping", None)
                joint.attrib.pop("frictionloss", None)

        for geom in root.iter("geom"):
            if geom.attrib.get("solref") == "0.0001 1.0":
                geom.set("solref", "0.02 1.0")


# -----------------------------------------------------------------
# 2. CREATE THE XML-GENERATING FUNCTION
# -----------------------------------------------------------------
def create_scene_xml(
        object_id,
        backend       = "mujoco",
        template_path = str(ROBOT_ASSETS_DIR / "scene_template.xml"),
        out           = str(ROBOT_ASSETS_DIR / "generated_scene.xml")
    ):
    return GeneratedSceneBuilder(backend=backend).write(object_id=object_id, out=out)


def create_object_xml(
        object_id,
        backend = "mujoco",
        out     = None,
    ):
    return GeneratedSceneBuilder(backend=backend).write_object(object_id=object_id, out=out)


def create_robot_xml(
        backend = "mujoco",
        out     = None,
    ):
    return GeneratedSceneBuilder(backend=backend).write_robot(out=out)


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
        nums            =[0, 10, 11, 12, 13, 14],
        template_path   = str(ROBOT_ASSETS_DIR / "scene_template.xml"),
        out             = str(ROBOT_ASSETS_DIR / "generated_scene.xml")
    ):
    asset_block = ""

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



def load_photoshoot(nums=[0, 10, 11, 12, 13, 14], launch_viewer=True):
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
