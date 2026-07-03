from __future__ import annotations

import os

from nyssa_bench.cli import main as nyssa_main


def main() -> int:
    os.environ.setdefault("NYSSA_BC_CHECKPOINT", "checkpoints/bc_policy.json")
    return nyssa_main(
        [
            "experiment",
            "--suite",
            "maniskill_manipulation_v0",
            "--engine",
            "maniskill",
            "--policies",
            "random",
            "scripted_oracle",
            "bc_policy",
            "--seeds",
            "0",
            "1",
            "2",
            "--episodes",
            "100",
            "--out",
            "benchmark_results/maniskill_manipulation_v0",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
