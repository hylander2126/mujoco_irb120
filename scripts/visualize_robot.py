#!/usr/bin/env python3
"""Open a minimal interactive viewer for robot frame inspection only.

Usage examples:
	python scripts/visualize_robot.py
	python scripts/visualize_robot.py --frame body
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "src" / "mujoco_irb120" / "assets"
COMMON_XML = (ASSETS_DIR / "common_modified.xml").as_posix()
ROBOT_XML = (ASSETS_DIR / "my_objects" / "robot" / "robot.xml").as_posix()


def _robot_only_xml() -> str:
	return f"""
<mujoco>
	<option timestep="0.001" />
	<include file="{COMMON_XML}"/>
	<worldbody>
		<include file="{ROBOT_XML}"/>
		<light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"
			   castshadow="false" pos="0 0 3" dir="0 0 -1" name="light0"/>
	</worldbody>
</mujoco>
"""


def main() -> None:
	parser = argparse.ArgumentParser(description="Minimal robot-only frame viewer")
	args = parser.parse_args()

	model = mujoco.MjModel.from_xml_string(_robot_only_xml())
	data = mujoco.MjData(model)

	with mujoco.viewer.launch_passive(model, data, show_left_ui=True) as viewer:
		# viewer.opt.frame = FRAME_MODE[args.frame]
		print(f"Viewer running (robot-only). Close window to exit.")

		while viewer.is_running():
			mujoco.mj_step(model, data)
			viewer.sync()
			time.sleep(model.opt.timestep)


if __name__ == "__main__":
	main()
