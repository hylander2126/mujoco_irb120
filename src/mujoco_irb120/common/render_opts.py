import mujoco
import numpy as np
import contextlib


### Camera parameters for BOTH viewer and offscreen renderer ###

CAM_DISTANCE   = 1.5 # Zoom level
CAM_ELEVATION  = -30 # Camera elevation angle
CAM_AZIMUTH    = 90 # Camera azimuth angle
CAM_LOOKAT     = np.array([0.75, 0, 0.25]) # structure: (x, y, z)



class RendererViewerOpts:
    def __init__(self, model_obj, data_obj, vis=True, width=1280, height=720, framerate=60):

        self.model_obj  = model_obj
        self.data_obj   = data_obj
        self.vis        = bool(vis)
        self.width      = int(width)
        self.height     = int(height)
        self.framerate  = int(framerate)

        # Initialize camera and visualization options
        self.cam_obj = mujoco.MjvCamera() # This will be our camera for rendering
        self.opt_obj = mujoco.MjvOption() # This will be our visualization options
        mujoco.mjv_defaultCamera(self.cam_obj)
        mujoco.mjv_defaultOption(self.opt_obj)

        # Video recording needs a renderer (as opposed to the passive viewer for live-viewing)
        self.renderer = mujoco.Renderer(model_obj, height=self.height, width=self.width)

        # Launch the viewer context if visualization is enabled
        self._viewer_ctx = (
            mujoco.viewer.launch_passive(model_obj, data_obj, show_left_ui=False) 
            if vis else contextlib.nullcontext(None)
        )
        self.viewer = None # becomes the actual viewer after __enter__

        # Frame buffer
        self.frames = []

        # Apply common model visualization scales
        self._apply_model_vis(self.model_obj)

        # Apply offscreen (Renderer) visualization options
        self._apply_offscreen_opts(self.cam_obj, self.opt_obj)


    # ---------------------- context manager ----------------------
    def __enter__(self):
        self.viewer = self._viewer_ctx.__enter__()  # will be None if vis=False
        if self.viewer is not None:
            self._apply_viewer_opts(self.viewer)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Close renderer reliably
        if getattr(self, "renderer", None) is not None:
            self.renderer.close()
            self.renderer = None
        # Exit viewer context (no-op if headless)
        return self._viewer_ctx.__exit__(exc_type, exc, tb)
    
    
    # ---------------------- public helpers ----------------------
    def viewer_is_running(self):
        return self.viewer.is_running() if self.viewer is not None else True

    def sync(self):
        if self.viewer is not None:
            self.viewer.sync()

    def capture_frame_if_due(self, data_obj):
        if len(self.frames) < data_obj.time * self.framerate:     # Add frame to the video recording
            self.renderer.update_scene(data_obj, camera=self.cam_obj, scene_option=self.opt_obj)  # Update the renderer with the current scene
            self.frames.append(self.renderer.render())        # Capture the current frame for video recording

    @staticmethod
    def _apply_viewer_opts(v_ctx):
        """ Set visualization options for the passive viewer context """
        v_ctx.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]    = True # Contact arrows
        # v_ctx.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]    = True # Gross... Contact 'translucent' force 'disc'
        
        # v_ctx.opt.frame                                          = mujoco.mjtFrame.mjFRAME_SITE # Visualize SITE frames only
        # v_ctx.opt.frame                                          = mujoco.mjtFrame.mjFRAME_BODY # Visualize BODY frames only
        # v_ctx.opt.frame                                          = mujoco.mjtFrame.mjFRAME_WORLD # Visualize WORLD frames only

        # v_ctx.opt.flags[mujoco.mjtVisFlag.mjVIS_COM]              = True # Center of mass spheres

        v_ctx.cam.distance                                       = CAM_DISTANCE
        v_ctx.cam.elevation                                      = CAM_ELEVATION
        v_ctx.cam.azimuth                                        = CAM_AZIMUTH
        v_ctx.cam.lookat[:]                                      = CAM_LOOKAT

    @staticmethod
    def _apply_model_vis(model_obj):
        model_obj.vis.scale.contactwidth    = 0.025  # Contact arrow width
        model_obj.vis.scale.contactheight   = 0.25   # Contact arrow height
        # model_obj.vis.scale.forcewidth      = 0.05 # Gross... force 'disc' size
        # model_obj.vis.map.force             = 0.3  # Gross... force 'disc' scale
        model_obj.vis.scale.framewidth      = 0.025  # Frame axis width
        model_obj.vis.scale.framelength     = 0.75   # Frame axis length

    @staticmethod
    def _apply_offscreen_opts(cam_obj, opt_obj):
        """ Set visualization options for the offscreen renderer (video recording) """
        opt_obj.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]    = True # Contact arrows
        # opt_obj.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]    = True # Contact 'translucent' force 'disc'
        # opt_obj.frame                                          = mujoco.mjtFrame.mjFRAME_BODY # Visualize BODY frames only
        opt_obj.frame                                          = mujoco.mjtFrame.mjFRAME_SITE # Visualize SITE frames only NOT WORKING TODO

        cam_obj.distance                                       = CAM_DISTANCE
        cam_obj.elevation                                      = CAM_ELEVATION
        cam_obj.azimuth                                        = CAM_AZIMUTH
        cam_obj.lookat[:]                                      = CAM_LOOKAT