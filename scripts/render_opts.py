import mujoco
import numpy as np

def set_viewer_opts(model_obj, viewer):
    # === Tweak scales of contact visualization elements ===
    model_obj.vis.scale.contactwidth = 0.025
    model_obj.vis.scale.contactheight = 0.25
    model_obj.vis.scale.forcewidth = 0.05
    model_obj.vis.map.force = 0.3
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True # joint viz
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE # Show site frame(s)
    # === Make site frame arrows smaller ===
    # model_obj.vis.scale.framewidth = 0.025
    # model_obj.vis.scale.framelength = .75
    # === Set default camera position ===
    viewer.cam.distance = 1.5       # Distance from the camera to the scene
    viewer.cam.elevation = -10#-30.0    # y-axis rotation
    viewer.cam.azimuth = 90#100.0      # z-axis rotation
    viewer.cam.lookat[:] = np.array([1, 0.0, 0.1])  # Center of the scene


# The set_render_opts function as it was defined, applied to mujoco.viewer
# For mujoco.Renderer, we will configure MjvCamera and MjvOption directly
# and pass them to update_scene.
def set_renderer_opts(model_obj, cam_obj, opt_obj):
    # === Tweak scales of contact visualization elements (these apply to model.vis) ===
    model_obj.vis.scale.contactwidth = 0.025
    model_obj.vis.scale.contactheight = 0.25
    model_obj.vis.scale.forcewidth = 0.05
    model_obj.vis.map.force = 0.3
    model_obj.vis.scale.framewidth = 0.025
    model_obj.vis.scale.framelength = .75

    # === Configure MjvOption flags ===
    opt_obj.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    opt_obj.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # === To show frames, you typically enable a flag like mjVIS_JOINT, mjVIS_BODY, mjVIS_GEOM, mjVIS_SITE, etc.
    # and then set the specific frame type using model.vis.frame ===
    opt_obj.frame = mujoco.mjtFrame.mjFRAME_SITE # Show site frame(s)


    # === Configure MjvCamera ===
    cam_obj.distance = 1.5       # Distance from the camera to the scene
    cam_obj.elevation = -30.0    # y-axis rotation
    cam_obj.azimuth = 100.0      # z-axis rotation
    cam_obj.lookat[:] = np.array([1, 0.0, 0.1])  # Center of the scene