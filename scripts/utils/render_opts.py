import mujoco
import numpy as np

def set_viewer_opts(model_obj, viewer):
    model_obj.vis.scale.contactwidth = 0.025
    model_obj.vis.scale.contactheight = 0.25
    model_obj.vis.scale.forcewidth = 0.05
    model_obj.vis.map.force = 0.3
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD | mujoco.mjtFrame.mjFRAME_SITE
    model_obj.vis.scale.framewidth = 0.025
    model_obj.vis.scale.framelength = 0.75
    viewer.cam.distance = 1.5
    viewer.cam.elevation = -10
    viewer.cam.azimuth = 90
    viewer.cam.lookat[:] = np.array([
        1,
        0,
        0.1])


def set_renderer_opts(model_obj, cam_obj, opt_obj):
    model_obj.vis.scale.contactwidth = 0.025
    model_obj.vis.scale.contactheight = 0.25
    model_obj.vis.scale.forcewidth = 0.05
    model_obj.vis.map.force = 0.3
    model_obj.vis.scale.framewidth = 0.025
    model_obj.vis.scale.framelength = 0.75
    opt_obj.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    opt_obj.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    opt_obj.frame = mujoco.mjtFrame.mjFRAME_SITE
    cam_obj.distance = 1.5
    cam_obj.elevation = -30
    cam_obj.azimuth = 100
    cam_obj.lookat[:] = np.array([
        1,
        0,
        0.1])
