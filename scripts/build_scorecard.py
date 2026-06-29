from __future__ import annotations

import argparse

from nyssa_bench.reports.scorecard import write_scorecard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a NyssaBench scorecard from real run directories.")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default="benchmark_results/baselines_v0.json")
    parser.add_argument("--benchmark", default="NyssaBench v0 baselines")
    parser.add_argument("--date")
    parser.add_argument("--comparison-out", default="reports/real_baselines_v0.html")
    parser.add_argument("--leaderboard-out", default="site/leaderboard/leaderboard.json")
    args = parser.parse_args(argv)

    paths = write_scorecard(
        args.runs,
        out=args.out,
        benchmark=args.benchmark,
        scorecard_date=args.date,
        comparison_report=args.comparison_out,
        leaderboard=args.leaderboard_out,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
