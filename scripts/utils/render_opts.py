import mujoco
import numpy as np
import contextlib

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
    def _apply_viewer_opts(viewer_ctx):
        """ Set visualization options for the passive viewer context """
        viewer_ctx.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]    = True
        viewer_ctx.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]    = True
        viewer_ctx.opt.frame                                          = mujoco.mjtFrame.mjFRAME_SITE
        viewer_ctx.opt.frame                                          = mujoco.mjtFrame.mjFRAME_WORLD | mujoco.mjtFrame.mjFRAME_SITE
        viewer_ctx.cam.distance                                       = 1.5 # Zoom level
        viewer_ctx.cam.elevation                                      = -10
        viewer_ctx.cam.azimuth                                        = 90
        viewer_ctx.cam.lookat[:]                                      = np.array([0.75, 0, 0.25]) # structure: (x, y, z)

    @staticmethod
    def _apply_model_vis(model_obj):
        model_obj.vis.scale.contactwidth    = 0.025
        model_obj.vis.scale.contactheight   = 0.25
        model_obj.vis.scale.forcewidth      = 0.05
        model_obj.vis.map.force             = 0.3
        model_obj.vis.scale.framewidth      = 0.025
        model_obj.vis.scale.framelength     = 0.75

    @staticmethod
    def _apply_offscreen_opts(cam_obj, opt_obj):
        opt_obj.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]    = True
        opt_obj.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]    = True
        opt_obj.frame                                          = mujoco.mjtFrame.mjFRAME_SITE
        
        cam_obj.distance               = 1.5
        cam_obj.elevation              = -30
        cam_obj.azimuth                = 100
        cam_obj.lookat[:]              = np.array([1, 0, 0.1])