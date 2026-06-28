from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nyssa_bench import PolicyRunner, Suite


def main() -> int:
    suite = Suite.load("mujoco_control_v0")
    PolicyRunner(policy="random", engine="mujoco", episodes=3, seed=42, out="runs/demo").evaluate(suite)
    print("wrote runs/demo/report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
