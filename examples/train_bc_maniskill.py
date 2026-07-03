from __future__ import annotations

from pathlib import Path

from nyssa_bench.cli import _train_bc_from_episode_files


def main() -> int:
    demo_dir = Path("benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle")
    episode_files = [demo_dir / f"seed_{seed}" / "episodes.json" for seed in (0, 1, 2)]
    missing = [path for path in episode_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing demo episode files. Run `nyssa experiment --policies scripted_oracle` first: "
            + ", ".join(str(path) for path in missing)
        )
    out = _train_bc_from_episode_files([str(path) for path in episode_files], "checkpoints/bc_policy.json", feature_dim=256, ridge=1e-3)
    print(f"bc_checkpoint: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
