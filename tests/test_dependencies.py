from pathlib import Path
import tomllib


def test_dependency_extras_are_declared():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    expected = {
        "all",
        "cli",
        "dataset",
        "dev",
        "diffusion",
        "experimental",
        "lerobot",
        "maniskill",
        "mujoco",
        "reports",
        "robomimic",
        "robocasa",
        "video",
        "vla",
        "full",
    }
    assert expected.issubset(extras)
