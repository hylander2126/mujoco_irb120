import mujoco
import numpy as np
import contextlib
from pathlib import Path
from pynput import keyboard as pynput_keyboard


### Camera parameters for BOTH viewer and offscreen renderer ###

CAM_DISTANCE   = 0.739 # 1.5 # Zoom level
CAM_ELEVATION  = -27.5 # 30 # Camera elevation angle
CAM_AZIMUTH    = 80.8 # 90 # Camera azimuth angle
CAM_LOOKAT     = np.array([0.563, -0.028, 0.208])# np.array([0.75, 0, 0.25]) # structure: (x, y, z)



class RendererViewerOpts:
    def __init__(self, model_obj, data_obj, vis=True, width=1280, height=720, framerate=60, show_left_UI=False):

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

        # Keyboard state: tracks which arrow keys are currently held down (via pynput)
        self.key_state = {
            pynput_keyboard.Key.left:  False,
            pynput_keyboard.Key.right: False,
            pynput_keyboard.Key.up:    False,
            pynput_keyboard.Key.down:  False,
        }
        self._kb_listener = None  # started in __enter__

        # Launch the viewer context if visualization is enabled
        self._viewer_ctx = (
            mujoco.viewer.launch_passive(model_obj, data_obj, show_left_ui=show_left_UI)
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
        self._kb_listener = pynput_keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._kb_listener.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._kb_listener is not None:
            self._kb_listener.stop()
            self._kb_listener = None
        if getattr(self, "renderer", None) is not None:
            self.renderer.close()
            self.renderer = None
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

    def save_video(self, path: str):
        """Write captured frames to an mp4 file using mediapy (pip install mediapy).

        No-ops gracefully if no frames were captured or mediapy is not installed.
        """
        if not self.frames:
            print("[RendererViewerOpts] No frames captured — skipping video save.")
            return
        try:
            import mediapy
        except ImportError:
            print("[RendererViewerOpts] mediapy not installed — cannot save video. "
                  "Run: pip install mediapy")
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        mediapy.write_video(path, self.frames, fps=self.framerate)
        print(f"[RendererViewerOpts] Video saved to {path}  ({len(self.frames)} frames @ {self.framerate} fps)")

    def _on_key_press(self, key):
        if key in self.key_state:
            self.key_state[key] = True

    def _on_key_release(self, key):
        if key in self.key_state:
            self.key_state[key] = False

    def get_keyboard_input(self):
        """Poll keyboard input and return velocity commands for cartesian control.

        Arrow keys control end-effector motion:
        - LEFT arrow:  Move -X (away from object)
        - RIGHT arrow: Move +X (toward object)
        - UP arrow:    Move +Z (lift)
        - DOWN arrow:  Move -Z (lower)

        Velocity is applied for as long as the key is held; stops when released.

        Returns:
            np.ndarray: 6D command [wx, wy, wz, vx, vy, vz] (rad/s and m/s)
                        or None if viewer is not available
        """
        if self.viewer is None or not self.viewer.is_running():
            return None

        v_cmd = np.zeros(6)
        max_lin_vel = 0.1  # m/s

        if self.key_state[pynput_keyboard.Key.right]:
            v_cmd[3] = max_lin_vel
        elif self.key_state[pynput_keyboard.Key.left]:
            v_cmd[3] = -max_lin_vel

        if self.key_state[pynput_keyboard.Key.up]:
            v_cmd[5] = max_lin_vel
        elif self.key_state[pynput_keyboard.Key.down]:
            v_cmd[5] = -max_lin_vel

        return v_cmd

    @staticmethod
    def _apply_viewer_opts(v_ctx):
        """ Set visualization options for the passive viewer context """
        # v_ctx.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]    = True # Contact arrows
        v_ctx.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]    = True # Gross... Contact 'translucent' force 'disc'
       # Just use keyboard shortcut ('v') to toggle visualization of frames as needed; too cluttered otherwise.
        # v_ctx.opt.flags[mujoco.mjtVisFlag.mjVIS_COM]              = True # Center of mass spheres TOO BIG, don't use...

        v_ctx.cam.distance                                       = CAM_DISTANCE
        v_ctx.cam.elevation                                      = CAM_ELEVATION
        v_ctx.cam.azimuth                                        = CAM_AZIMUTH
        v_ctx.cam.lookat[:]                                      = CAM_LOOKAT

    @staticmethod
    def _apply_model_vis(model_obj):
        model_obj.vis.scale.contactwidth    = 0.004  # Contact arrow width
        model_obj.vis.scale.contactheight   = 0.02   # Contact arrow height
        # model_obj.vis.scale.forcewidth      = 0.05 # Gross... force 'disc' size
        # model_obj.vis.map.force             = 0.3  # Gross... force 'disc' scale
        model_obj.vis.scale.framewidth      = 0.025  # Frame axis width
        model_obj.vis.scale.framelength     = 0.75   # Frame axis length

    @staticmethod
    def _apply_offscreen_opts(cam_obj, opt_obj):
        """ Set visualization options for the offscreen renderer (video recording) """
        # opt_obj.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]    = True # Contact arrows
        opt_obj.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]    = True # Contact 'translucent' force 'disc'
        # opt_obj.frame                                          = mujoco.mjtFrame.mjFRAME_BODY # Visualize BODY frames only
        # opt_obj.frame                                          = mujoco.mjtFrame.mjFRAME_SITE # Visualize SITE frames only NOT WORKING TODO

        cam_obj.distance                                       = CAM_DISTANCE
        cam_obj.elevation                                      = CAM_ELEVATION
        cam_obj.azimuth                                        = CAM_AZIMUTH
        cam_obj.lookat[:]                                      = CAM_LOOKAT