from pathlib import Path
import argparse
import json

import mujoco
import numpy as np

from mujoco_irb120.common.load_obj_in_env import load_environment
import mujoco_irb120.common.robot_controller as robot_controller


def summarize(name: str, arr: np.ndarray) -> None:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    print(f"\n[{name}]")
    print(f"  mean force  : {mean[:3]}")
    print(f"  mean torque : {mean[3:]}")
    print(f"  std force   : {std[:3]}")
    print(f"  std torque  : {std[3:]}")


def collect_stationary_samples(irb, model, data, q_hold, n_samples: int) -> dict:
    w_raw_site = []
    w_world_no_gc_no_bias = []
    w_world_gc_no_bias = []
    w_world_gc_with_bias = []
    contact_flags = []

    for _ in range(n_samples):
        irb.set_pos_ctrl(q_hold, check_ellipsoid=False)
        mujoco.mj_step(model, data)

        f_meas = np.asarray(data.sensordata[irb.f_adr:irb.f_adr + 3], dtype=float)
        t_meas = np.asarray(data.sensordata[irb.t_adr:irb.t_adr + 3], dtype=float)
        w_site = np.concatenate([f_meas, t_meas])

        w_raw_site.append(w_site)
        w_world_no_gc_no_bias.append(irb.ft_get_reading(grav_comp=False, apply_bias=False))
        w_world_gc_no_bias.append(irb.ft_get_reading(grav_comp=True, apply_bias=False))
        w_world_gc_with_bias.append(irb.ft_get_reading(grav_comp=True, apply_bias=True))
        contact_flags.append(irb.check_contact())

    return {
        "raw_site": np.asarray(w_raw_site),
        "world_no_gc_no_bias": np.asarray(w_world_no_gc_no_bias),
        "world_gc_no_bias": np.asarray(w_world_gc_no_bias),
        "world_gc_with_bias": np.asarray(w_world_gc_with_bias),
        "contact": np.asarray(contact_flags, dtype=bool),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug FT pipeline in stationary no-contact conditions.")
    parser.add_argument("--object", type=int, default=0, help="Object id from object_params.json")
    parser.add_argument("--settle", type=int, default=250, help="Settle steps before measurement")
    parser.add_argument("--samples", type=int, default=400, help="Samples per test phase")
    parser.add_argument("--bias-samples", type=int, default=200, help="Samples used by ft_bias")
    parser.add_argument("--plot", action="store_true", help="Plot x/y/z traces for quick visual verification")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    params_path = repo_root / "scripts" / "object_params.json"

    with open(params_path, "r", encoding="utf-8") as f:
        obj_params = json.load(f)["objects"][str(args.object)]

    init_xyz = np.array(obj_params["init_xyz"], dtype=float)

    model, data = load_environment(num=args.object, launch_viewer=False)
    irb = robot_controller.controller(model, data)

    t_home = irb.FK()
    t_init = t_home.copy()
    t_init[:3, 3] = init_xyz

    q_init = irb.IK(t_init, method=2, damping=0.5, max_iters=1000)
    irb.set_pose(q=q_init)

    print("--- Stationary FT Debug ---")
    print(f"object id: {args.object}")
    print(f"grav_mass used in controller: {irb.grav_mass}")
    print(f"model gravity vector: {model.opt.gravity}")

    for _ in range(args.settle):
        irb.set_pos_ctrl(q_init, check_ellipsoid=False)
        mujoco.mj_step(model, data)

    # Phase 1: Before bias
    phase1 = collect_stationary_samples(irb, model, data, q_init, args.samples)
    print(f"\nPhase 1 contact fraction: {phase1['contact'].mean():.4f}")
    summarize("raw_site", phase1["raw_site"])
    summarize("world_no_gc_no_bias", phase1["world_no_gc_no_bias"])
    summarize("world_gc_no_bias", phase1["world_gc_no_bias"])

    gc_delta = phase1["world_no_gc_no_bias"] - phase1["world_gc_no_bias"]
    summarize("gravity_comp_delta(no_gc - gc)", gc_delta)

    # Phase 2: Bias
    print("\nRunning bias routine (script-side deterministic bias)...")
    bias_samples = []
    for _ in range(args.bias_samples):
        irb.set_pos_ctrl(q_init, check_ellipsoid=False)
        mujoco.mj_step(model, data)
        bias_samples.append(irb.ft_get_reading(grav_comp=True, apply_bias=False))

    b = np.mean(np.asarray(bias_samples, dtype=float), axis=0)

    # Apply to whichever field this controller version uses.
    if hasattr(irb, "ft_bias_val"):
        irb.ft_bias_val = b.copy()
        bias_field = "ft_bias_val"
    elif hasattr(irb, "ft_offset"):
        irb.ft_offset = b.copy()
        bias_field = "ft_offset"
    else:
        bias_field = "(none found)"

    print(f"stored bias field: {bias_field}")
    print(f"stored bias value: {b}")

    # Phase 3: After bias
    phase3 = collect_stationary_samples(irb, model, data, q_init, args.samples)
    print(f"\nPhase 3 contact fraction: {phase3['contact'].mean():.4f}")
    summarize("world_gc_with_bias", phase3["world_gc_with_bias"])

    # Forward-only check: confirms viewer is irrelevant and dynamics stepping is the source of any variance.
    w_forward_only = []
    for _ in range(100):
        mujoco.mj_forward(model, data)
        w_forward_only.append(irb.ft_get_reading(grav_comp=True, apply_bias=True))
    w_forward_only = np.asarray(w_forward_only)
    summarize("forward_only_gc_with_bias", w_forward_only)

    if args.plot:
        import matplotlib.pyplot as plt

        t = np.arange(args.samples) * model.opt.timestep
        names = ["f_x", "f_y", "f_z"]

        fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for i in range(3):
            ax[i].plot(t, phase1["world_gc_no_bias"][:, i], label="pre-bias (gc only)", linewidth=2)
            ax[i].plot(t, phase3["world_gc_with_bias"][:, i], label="post-bias (gc+bias)", linewidth=2)
            ax[i].set_ylabel(f"{names[i]} (N)")
            ax[i].grid(True)
            ax[i].legend(loc="best")
        ax[-1].set_xlabel("time (s)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
