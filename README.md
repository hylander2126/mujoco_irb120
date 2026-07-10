# mujoco_irb120

MuJoCo/Genesis model and controllers for the ABB IRB120 6-DOF manipulator. This repo is meant to be used as a submodule: it holds the robot's MJCF model, meshes, a handful of hand-made test objects, and the generic controller code used to drive it — no experiment- or research-specific logic.

---

## Repository Structure

```
mujoco_irb120/
├── robot/
│   ├── assets/
│   │   ├── robot/                 # IRB120 MuJoCo model + visual meshes
│   │   ├── objects/                # Hand-made test objects (box, heart, flashlight, L, monitor, soda)
│   │   └── genesis_object.xml      # Standalone object MJCF used by the Genesis demo
│   ├── controllers/
│   │   ├── robot.py                # MuJoCo controller: IK/FK, Jacobians, force/torque sensing, gravity compensation
│   │   └── genesis_robot.py        # Genesis controller: guarded Cartesian velocity shove, workspace/manipulability fade-out
│   └── __init__.py
├── util/
│   └── helper_fns.py               # Rotation/screw-theory math (SO3/SE3, quaternions, Jacobians) used by robot.py
└── scripts/
    └── genesis_test.py             # Smoke test: loads the IRB120 + a box into Genesis and runs a scripted shove
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the Genesis smoke test from the repo root:

```bash
python scripts/genesis_test.py --show-viewer true
```

or headless (saves a video to `outputs/genesis_test.mp4`):

```bash
python scripts/genesis_test.py --show-viewer false
```

### Controllers

- `robot.controllers.robot.controller` — MuJoCo-side controller. Wraps a `mujoco.MjModel`/`mujoco.MjData` pair, exposing `FK`/`IK` (Newton-Raphson, damped least squares, or gradient descent), Jacobian-based velocity control, force/torque sensor readings with gravity compensation, and contact/topple checks against a payload body.
- `robot.controllers.genesis_robot.GenesisRobotController` — Genesis-side controller. Wraps a Genesis entity + scene, adding a guarded `velocity_shove` primitive (trapezoidal speed profile, workspace ellipsoid fade-out, manipulability fade-out, damped-least-squares joint velocity) on top of Genesis's built-in IK/planning.

### Robot Model — ABB IRB120

| Parameter | Value |
|-----------|-------|
| DOF | 6 revolute joints |
| Payload | 3 kg |
| Reach | 580 mm |
| J1 range | ±165° |
| J2 range | ±110° |
| J3 range | ±70° |
| J4 range | ±160° |
| J5 range | ±120° |
| J6 range | ±400° |
| Force/torque sensor offset | 82.25 mm from flange |
| Pusher finger length | 110 mm |

Control gains: kp = 200 / kv = 100 (joints 1–3); kp = 100 / kv = 50 (joints 4–6)
