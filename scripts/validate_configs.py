from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nyssa_bench.core.suite import list_suites, Suite
from nyssa_bench.core.task import TaskSpec, list_tasks


def main() -> int:
    for suite_id in list_suites():
        Suite.load(suite_id)
    for task_id in list_tasks():
        TaskSpec.load(task_id)
    print(f"valid suites={len(list_suites())} tasks={len(list_tasks())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
