# Hardware tool snapshot

Source: `irb120_ros2/irb120_control`, commit
`83ee38617f9ef30b077074c0072a1c289407aa74`. `manifest.json` records SHA-256
hashes of the actual URDF/xacro/glTF inputs, including any local changes.

Regenerate from the parent simulation repository:

```bash
.venv/bin/python scripts/sync_hardware_tool.py ../irb120_ros2
```

The six original glTF meshes and three assembly definitions are retained here.
STL conversion preserves mesh vertices and URDF visual transforms. Meshes are
baked into flange-aligned coordinates; finger meshes are relative to the finger
mounting face. The shared `robot.xml` tool subtree is generated from this chain.
`meshes.xml` declares assets relative to the shared MuJoCo assets directory.

Compared with the previous model:

- Finger root X: 82.250 -> 85.8244 mm from flange.
- Ball center X: 179.000 -> 172.4756 mm from flange.
- Actual chain includes approximately -0.502 mm Z offset.
- Ball radius remains 13.25 mm.
- Tool-stack collision cylinder: 75 -> 90 mm diameter, 82.25 -> 85.8 mm length.
- Finger mass: 64 -> measured 66 g; measured CoG: 27.5 mm along finger axis.
  The rotational inertia uses the finger URDF CAD tensor. That tensor is not
  re-estimated from the measured CoG/mass.

Physical simplifications retained: sensor/adapter meshes are visual-only and
massless, covered by the hardware's conservative collision cylinder. The finger
uses analytic sphere contact; the shaft remains visual-only. The experiments'
existing policy excludes adapter/object contact, but retains adapter/table
contact. This is not a full hardware collision/inertia calibration. The robot
base, simulation table/world placement, and separate Genesis model are unchanged.

The force site lies at the finger mounting face. Its simulation-local axes remain
unchanged; the controller rotates the wrench to world coordinates. MuJoCo's
parent-on-child reaction wrench is gravity-compensated using the measured pusher
CoG before conversion to external wrench sign. Hardware still references the
removed `ft_link` TF in places, and its estimator hard-codes the old 82.25 mm
stack offset. Those ROS files were inspected but not modified here; the actual
current URDF chain determines this simulation's geometry.
