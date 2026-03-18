# MuJoCo IRB120 — 3D Center-of-Mass Estimation via Robotic Tipping

A physics-based framework for estimating the 3D center-of-mass (CoM) of unknown objects using an ABB IRB120 6-DOF manipulator. The robot pushes objects to the edge of a table, and a torque-balance model is fit to the observed tipping dynamics to recover object mass and CoM height. Both MuJoCo simulation and real-world experiments on a physical robot are supported.

---

## Demo

| Simulation & Experiment Results |
|:---:|
| ![Simulation and Experiment Comparison](figures/sim_and_exp_topple.png) |

| Object Lineup | Force/Torque Signals (Sim) | Force/Torque Signals (Exp) |
|:---:|:---:|:---:|
| ![Objects](figures/sim_objects.png) | ![Sim XYZ](figures/sim_topple_xyz.png) | ![Exp XYZ](figures/exp_topple_xyz.png) |

| Model Fit — Simulation | Model Fit — Experiment |
|:---:|:---:|
| ![Sim Fit](figures/sim_topple_fit25.png) | ![Exp Fit](figures/exp_topple_fit25.png) |

---

## Overview

### Problem Statement

Estimating an object's center of mass from robot interactions is a fundamental capability for manipulation tasks such as grasping, transportation, and placement. This project addresses **CoM estimation without prior knowledge of object geometry**, using only the measured contact forces and object rotation during a controlled tipping maneuver.

### Approach

1. **Tipping Maneuver**: The robot pushes an object horizontally until it begins to tip over the table edge. A 6-axis force/torque sensor on the end-effector records contact forces throughout the motion.
2. **Orientation Tracking**: An AR tag on the object provides rotation angle θ via quaternion estimates (physical experiments), or the MuJoCo joint state is used directly (simulation).
3. **Torque-Balance Model**: At each timestep, the applied torque τ_app = r_f × F is matched against the gravity torque τ_grav = −r_c × R(−θ) · W using nonlinear least squares. The free parameters are mass m and CoM height z_c.
4. **Tipping Angle Prediction**: The estimated CoM is used to compute the critical tipping angle θ* = arctan(d_xy / z_c), which is the angle at which the object becomes unstable.

---

## Repository Structure

```
mujoco_irb120/
├── analyze_experiment.py          # Main analysis pipeline for experiment CSV data
├── scripts/
│   ├── main.ipynb                 # Simulation notebook (robot control + data collection)
│   ├── load_obj_in_env.py         # Procedurally generates MuJoCo XML scenes
│   ├── test_ik.py                 # Standalone IK solver validation
│   ├── photoshoot.py              # Multi-object visualization scenes
│   └── utils/
│       ├── robot_controller.py    # IK/FK, force sensing, gravity compensation
│       ├── com_estimation.py      # Physics models (τ, F) and curve fitting
│       └── helper_fns.py          # Rotation utilities (quaternions, axis-angle, SO3)
├── assets/
│   ├── my_objects/                # Custom test object meshes + IRB120 robot URDF/XML
│   │   └── robot/                 # ABB IRB120 MuJoCo model
│   ├── object_sim/                # External object library (git submodule)
│   ├── _generated/                # Auto-generated scaled asset copies
│   ├── table_push.xml             # Base MuJoCo scene (table + robot)
│   └── common_modified.xml        # Shared rendering/material settings
├── experiments/                   # CSV data from physical robot trials (500 Hz)
├── figures/                       # Output plots and visualizations
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10+
- MuJoCo (installed automatically via the `mujoco` Python package)
- Git (for submodule initialization)

### Installation

**1. Clone with submodules** (required for object assets):

```bash
git clone --recurse-submodules https://github.com/<your-username>/mujoco_irb120.git
cd mujoco_irb120
```

If you already cloned without `--recurse-submodules`, initialize the submodule manually:

```bash
git submodule update --init --recursive
```

**2. Create a virtual environment** (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**:

```bash
pip install mujoco numpy scipy matplotlib mediapy
```

> **Note:** A `requirements.txt` is provided if available. You can also install directly:
> `pip install -r requirements.txt`

---

## Usage

### Analyze Experimental Data

Run the full analysis pipeline on the recorded CSV experiments:

```bash
python analyze_experiment.py
```

This will:
- Load experiment CSVs from `experiments/` for all test objects (box, heart, flashlight, monitor, soda)
- Apply a Butterworth + median + Savitzky-Golay filter cascade to the raw force signals
- Fit the torque-balance model to estimate mass *m* and CoM height *z_c*
- Compute tipping angle θ* and compare against ground-truth CAD values
- Save analysis plots to `figures/`

### Run the Simulation

Open the Jupyter notebook to simulate the robot performing the tipping maneuver:

```bash
cd scripts
jupyter notebook main.ipynb
```

The notebook covers:
- Loading an object into the MuJoCo scene
- Executing the robot push trajectory via damped-least-squares IK
- Recording force/torque, EE pose, and object angle data
- Running the CoM estimation pipeline on simulated data

### Test the IK Solver

```bash
cd scripts
python test_ik.py
```

Validates the damped-least-squares IK solver against known target poses and prints convergence statistics.

### Load Custom Objects

To generate a MuJoCo scene with a specific object:

```python
from load_obj_in_env import load_environment

model, data = load_environment(num=0)   # 0=box, 10=heart, 11=L-shape,
                                        # 12=monitor, 13=soda, 14=flashlight
```

---

## Technical Details

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

### Inverse Kinematics

Three solvers are implemented in `robot_controller.py`:

| Method | Description |
|--------|-------------|
| Newton-Raphson | Standard Jacobian-based iteration |
| **Damped Least Squares** (default) | J^T (JJ^T + λ²I)^{-1} — robust near singularities |
| Gradient Descent | Step-based minimization of pose error |

The damping coefficient λ is adjusted dynamically based on the convergence rate to balance accuracy and singularity avoidance.

### CoM Estimation — Physics Model

The tipping maneuver produces a time series of contact force **F**(t) and object rotation angle θ(t). The model parameters m (mass) and z_c (CoM height) are recovered by minimizing:

```
min_{m, z_c}  ||τ_app(t) − τ_grav(θ(t), m, z_c)||²
```

where:
- **τ_app** = r_f × F — applied torque from finger contact
- **τ_grav** = −r_c × R(−θ) · W — gravity torque about the tipping edge
- **r_c** = [d_x, d_y, z_c]ᵀ — CoM position vector from pivot
- **R(−θ)** — rotation matrix for object angle about the tipping axis

Fitting is performed with `scipy.optimize.curve_fit` (Levenberg-Marquardt).

### Signal Processing

Raw force signals are passed through a three-stage filter cascade:

1. **Butterworth low-pass** — 4th order, 5 Hz cutoff (removes high-frequency vibration)
2. **Median filter** — kernel size 5 (removes impulse artifacts)
3. **Savitzky-Golay** — 3rd order polynomial, kernel size 89 (smooths while preserving tipping transient shape)

### Test Objects

| Object | Ground Truth Mass | Ground Truth z_c | Notes |
|--------|:-----------------:|:----------------:|-------|
| Box | 664 g | 146 mm | Uniform rectangular prism |
| Heart | 236 g | 98 mm | Irregular geometry |
| Flashlight | 387 g | 97 mm | Cylindrical, off-axis mass |
| L-shape | 106 g | 58 mm | Asymmetric cross-section |
| Monitor | 5008 g | 252 mm | Large, heavy object |
| Soda can | 2071 g | 115 mm | Cylindrical, near-uniform |

### Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Timestep | 1 ms (1 kHz) |
| Integrator | Implicit-fast |
| Gravity | 9.81 m/s² |
| Table friction | μ = 1.0 |
| Experiment sample rate | 500 Hz |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `mujoco` | Physics simulation and rendering |
| `numpy` | Numerical arrays and linear algebra |
| `scipy` | Signal filtering, nonlinear optimization, rotation utilities |
| `matplotlib` | Plotting and figure generation |
| `mediapy` | Video recording from MuJoCo renderer |

Object assets: [vikashplus/object_sim](https://github.com/vikashplus/object_sim)

---

## License

This project is for research and educational purposes. See [LICENSE](LICENSE) for details.
