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
ASSETS_DIR = REPO_ROOT / "robot" / "assets"
SCENE_TEMPLATE = ASSETS_DIR / "scene_template.xml"


def _robot_only_xml() -> str:
	template = SCENE_TEMPLATE.read_text()
	return template.format(
		actuator_block="",
		asset_block="",
		object_block='<include file="robot/robot.xml"/>',
	)


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
