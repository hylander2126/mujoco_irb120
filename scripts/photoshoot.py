import mujoco
import mujoco.viewer
from pathlib import Path

# ---------------------------------------------------------------------
# 1) EDIT THIS: list your object xmls + their approximate table z
#    - name: base name of your object xml (without .xml)
#    - z: height of the object frame so it sits flat on table
# ---------------------------------------------------------------------
OBJECTS = [
    {"name": "box",         "z": 0.10},  # example
    {"name": "heart",       "z": 0.10},  # example
    {"name": "L",           "z": 0.11},  # tweak as needed
    {"name": "monitor",     "z": 0.11},  # tweak as needed
    {"name": "soda",        "z": 0.11},  # tweak as needed
    {"name": "flashlight",  "z": 0.11},  # tweak as needed
]

# Directory where your object xmls live, relative to this script
OBJECT_DIR = Path("../assets/objects")      # <-- adjust if needed
SHARED_XML = Path("../assets/shared.xml")   # your shared/common xml
OUTPUT_XML = Path("photoshoot_scene.xml")


def make_photoshoot_xml():
    # Basic header + include your shared.xml for materials, etc.
    xml_parts = [
        '<mujoco>',
        f'  <include file="{SHARED_XML.as_posix()}" />',
        '  <worldbody>',
        '    <!-- Simple table / floor -->',
        '    <body name="table" pos="0 0 0">',
        '      <geom name="table_top" type="plane" size="2 2 0.1" pos="0 0 0.0" rgba="0.9 0.9 0.9 1"/>',
        '    </body>',
        '',
        '    <!-- Objects laid out left to right along +y -->'
    ]

    x = 0.7          # all objects at same x, in front of camera
    y_start = -0.25  # starting y
    dy = 0.25        # spacing between objects

    for i, obj in enumerate(OBJECTS):
        name = obj["name"]
        z = obj["z"]
        y = y_start + i * dy

        obj_xml_path = (OBJECT_DIR / f"{name}.xml").as_posix()

        xml_parts += [
            f'    <body name="{name}" pos="{x:.3f} {y:.3f} {z:.3f}" quat="1 0 0 0">',
            f'      <include file="{obj_xml_path}"/>',
            '    </body>',
            ''
        ]

    xml_parts += [
        '  </worldbody>',
        '</mujoco>'
    ]

    OUTPUT_XML.write_text("\n".join(xml_parts))
    print(f"Wrote {OUTPUT_XML}")


def main():
    make_photoshoot_xml()
    model = mujoco.MjModel.from_xml_path(OUTPUT_XML.as_posix())
    data = mujoco.MjData(model)

    # No physics needed; just render first frame
    with mujoco.viewer.launch_passive(model, data) as v:
        print("Photoshoot scene loaded. Adjust camera and screenshot.")
        while v.is_running():
            mujoco.mj_step(model, data)  # tiny steps just to keep viewer alive


if __name__ == "__main__":
    main()
