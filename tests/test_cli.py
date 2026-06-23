from pathlib import Path

from nyssa_bench.cli import main


def test_cli_lists_and_validates():
    assert main(["list-suites"]) == 0
    assert main(["list-tasks"]) == 0
    assert main(["list-engines"]) == 0
    assert main(["list-policies"]) == 0
    assert main(["validate", "tabletop_manipulation_v0"]) == 0
    assert main(["validate", "pick_cube"]) == 0


def test_cli_run_and_export(tmp_path: Path):
    run_dir = tmp_path / "run"
    other_run_dir = tmp_path / "other_run"

    assert main(
        [
            "run",
            "--suite",
            "warehouse_manipulation_v0",
            "--engine",
            "dummy",
            "--policy",
            "random",
            "--episodes",
            "1",
            "--out",
            str(run_dir),
        ]
    ) == 0
    assert (run_dir / "report.html").exists()

    assert main(["export", "--run", str(run_dir), "--format", "lerobot"]) == 0
    assert (run_dir / "lerobot" / "meta.json").exists()

    assert main(
        [
            "run",
            "--suite",
            "warehouse_manipulation_v0",
            "--engine",
            "dummy",
            "--policy",
            "diffusion",
            "--episodes",
            "1",
            "--out",
            str(other_run_dir),
        ]
    ) == 0
    assert main(["compare", str(run_dir), str(other_run_dir), "--out", str(tmp_path / "compare.html")]) == 0
    assert main(["leaderboard", str(run_dir), str(other_run_dir), "--out", str(tmp_path / "leaderboard.json")]) == 0
    assert (tmp_path / "compare.html").exists()
    assert (tmp_path / "leaderboard.json").exists()


def test_scripts_smoke():
    from scripts.release_checklist import main as release_checklist
    from scripts.validate_configs import main as validate_configs

    assert validate_configs() == 0
    assert release_checklist() == 0
