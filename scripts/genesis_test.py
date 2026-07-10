import argparse
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

# np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})
torch.set_printoptions(sci_mode=False)

REPO_ROOT = Path(__file__).resolve().parents[1] # since this is called from ./scripts/
CACHE_ROOT = Path("/tmp") / "mujoco_irb120-cache"
ROBOT_XML = REPO_ROOT / "robot" / "assets" / "robot" / "genesis_robot.xml"
OBJECT_XML = REPO_ROOT / "robot" / "assets" / "genesis_object.xml"
OUTPUT_DIR = REPO_ROOT / "outputs"
VIDEO_PATH = OUTPUT_DIR / "genesis_test.mp4"

os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_ROOT / "numba"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))

import genesis as gs
from parameter_estimation.genesis_robot import GenesisRobotController

def str_to_bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in ("true", "t", "yes", "y", "1"):
        return True
    if value in ("false", "f", "no", "n", "0"):
        return False

    raise argparse.ArgumentTypeError("Expected true or false.")


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test the IRB120 MJCF in Genesis.")
    parser.add_argument(
        "--show-viewer",
        type=str_to_bool,
        default=True,
        help="Open the live Genesis viewer.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of simulation steps to run.",
    )
    parser.set_defaults(show_viewer=True)
    return parser.parse_args()


def open_video(path):
    try:
        subprocess.Popen(
            ["xdg-open", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        print(f"Could not open video automatically: {exc}")


def main():
    args = parse_args()

    ###########################################################
    ########################## Setup ##########################
    ###########################################################

    # Keep the Genesis test intentionally simple: load the canonical robot and
    # object MJCF files directly, without generating a Genesis-specific scene.
    os.chdir(REPO_ROOT)

    gs.init(backend=gs.cpu)

    ############ Initialize scene options ############
    scene = gs.Scene(
        vis_options = gs.options.VisOptions(
            show_world_frame = True, # visualize the coordinate frame of `world` at its origin
            world_frame_size = 1.0, # length of the world frame in meter
            show_link_frame  = False, # do not visualize coordinate frames of entity links
            show_cameras     = False, # do not visualize mesh and frustum of the cameras added
            plane_reflection = True, # turn on plane reflection
            ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
        ),
        viewer_options = gs.options.ViewerOptions(
            res           = None, # (1280, 960),
            camera_pos    = (1.0, -1.0, 1.5),
            camera_lookat = (0.5, 0.0, 0.5),
            camera_fov    = 60,
            refresh_rate  = 60,
            enable_gui    = False, # True
        ),
        renderer          = gs.renderers.Rasterizer(), # using rasterizer for camera rendering
        show_viewer       = args.show_viewer,
    )

    video_camera = None
    if not args.show_viewer:
        video_camera = scene.add_camera(
            res=(1280, 720),
            pos=(1.0, -1.0, 1.0),
            lookat=(0.5, 0.0, 0.5),
            fov=60,
            GUI=False,
        )
    
    ############ Add scene primitives, IRB, and object ############
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(4.0, 4.0, 0.1),
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=[1.0, 1.0, 1.0], opacity=1.0),
        material=gs.materials.Rigid(friction=0.1, needs_coup=True),
    )

    irb = scene.add_entity(
        gs.morphs.MJCF(file=str(ROBOT_XML)),
    )

    obj = scene.add_entity(
        gs.morphs.MJCF(
            file=str(OBJECT_XML),
            # pos=(0.0, -0.08, 0.05),
            pos = (0.5, 0.16, 0.25),
            ),
        surface=gs.surfaces.Default(color=[1.0, 0.0, 0.0], opacity=1.0),
        material=gs.materials.Rigid(friction=0.1, needs_coup=True, rho=4500.0),
    )
    ########################## build ##########################

    if video_camera is not None:
        scene.start_recording(
            data_func = lambda: video_camera.render(rgb=True)[0],
            rec_options = gs.recorders.VideoFile(
                filename = "outputs/genesis_test.mp4",
                hz = 30,
            ),
        )

    scene.build()

    ###########################################################
    ######################### CONTROL #########################
    ###########################################################

    robot = GenesisRobotController(irb, scene)
    robot.configure_default_gains()
    robot.velocity_shove(
        preshove_pos = np.array([0.4, 0.08, 0.25]), # 0.11, 0.25, or 0.45 (low, mid, high). Centroid is 0.2 + 0.05 table height=0.25
        preshove_quat = np.array([1, 0, 0, 0]),
        push_direction = np.array([0.0, 1.0, 0.0]),
        shove_speed = 2.0,
        obj = obj,
        camera = video_camera,
        snap = True # TESTING WITHOUT MOTION PLANNING
    )

    if video_camera is not None:
        # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # video_camera.stop_recording(save_to_filename=str(VIDEO_PATH), fps=60)
        scene.stop_recording()
        print(f"Saved video to {VIDEO_PATH}")
        open_video(VIDEO_PATH)


if __name__ == "__main__":
    main()
