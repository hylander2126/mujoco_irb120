import argparse
import os
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

os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_ROOT / "numba"))

sys.path.insert(0, str(REPO_ROOT))

import genesis as gs
from robot.controllers.genesis_robot import GenesisRobotController

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
            max_FPS       = None, #60,
            enable_gui    = False, # True
        ),
        renderer          = gs.renderers.Rasterizer(), # using rasterizer for camera rendering
        show_viewer       = args.show_viewer,
    )
    
    ############ Add scene primitives, IRB, and object ############
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(4.0, 4.0, 0.1),
            fixed=True,
        ),
    )

    irb = scene.add_entity(
        gs.morphs.MJCF(file=str(ROBOT_XML)),
    )

    obj = scene.add_entity(
        gs.morphs.MJCF(
            file=str(OBJECT_XML),
            # pos=(0.3, 0.0, 0.05),
            pos = (0.0, 0.0, 0.0),
            ),
        surface=gs.surfaces.Default(color=[1.0, 0.0, 0.0], opacity=0.9),
        material=gs.materials.Rigid(friction=1.0, needs_coup=True),
    )
    ########################## build ##########################
    scene.build()



    ###########################################################
    ######################### CONTROL #########################
    ###########################################################

    robot = GenesisRobotController(irb, scene)
    robot.configure_default_gains()
    robot.velocity_shove(
        preshove_pos = np.array([0.30, 0.0, 0.18]),
        preshove_quat = np.array([1, 0, 0, 0]),
        push_direction = np.array([1.0, 0.0, 0.0]),
        obj = obj,
    )



## Deprecated, use robot controller instead.
def motion_test(irb, scene, dofs_idx, step):
    if step < 500:
        irb.set_dofs_position(np.array([1, 1, 0, 0, 0, 0]), dofs_idx) # make sudden changes to robot state without obeying physics
    elif step < 1000:
        irb.control_dofs_position(np.array([-1, 0.2, 1, -2, 1, 0.5]), dofs_idx) # use builtin PD POSITION control
    elif step < 1500:
        # Control first dof with VELOCITY and the rest with POSITION
        irb.control_dofs_position(np.array([0, 0, 0, 0, 0, 0])[1:], dofs_idx[1:])

        irb.control_dofs_velocity(np.array([1.0, 0, 0, 0, 0, 0])[:1], dofs_idx[:1]) # use builtin PD VELOCITY control
    else:
        irb.control_dofs_force(np.array([0, 0, 0, 0, 0, 0]), dofs_idx) # use builtin PD FORCE control
    
    if step % 100 == 0: # don't spam
        # This is control force computed based on given control scheme. If using force ctrl, equivalent to command.
        print('Control force: ', irb.get_dofs_control_force(dofs_idx))
        # This is actual force experienced by the dof.
        print('internal force: ', irb.get_dofs_force(dofs_idx))

    scene.step()
        

if __name__ == "__main__":
    main()
