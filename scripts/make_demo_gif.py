from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nyssa_bench.core.suite import Suite
from nyssa_bench.core.registry import make_engine
from nyssa_bench.policies.scripted_policy import ScriptedPolicy


def main() -> int:
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("Demo GIF generation requires: pip install -e '.[video]'") from exc

    suite = Suite.load("tabletop_manipulation_v0")
    task = suite.tasks[0]
    engine = make_engine("dummy")
    policy = ScriptedPolicy()
    frames = []

    try:
        engine.load_task(task)
        observation, _ = engine.reset(seed=42)
        frames.append(engine.render())
        for _ in range(18):
            observation, _, terminated, truncated, _ = engine.step(policy.act(observation))
            frames.append(engine.render())
            if terminated or truncated:
                break
    finally:
        engine.close()

    out = Path("docs/assets/nyssabench-demo.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frames, duration=90, loop=0)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
