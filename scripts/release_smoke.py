from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a clean NyssaBench release check.")
    parser.add_argument("--venv", default=".release-venv")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    venv = ROOT / args.venv
    if not args.skip_install:
        if venv.exists() and not args.keep:
            shutil.rmtree(venv)
        _run([sys.executable, "-m", "venv", str(venv)])
        _run([str(_python(venv)), "-m", "pip", "install", "--upgrade", "pip"])
        _run([str(_python(venv)), "-m", "pip", "install", "-e", ".[dev,video,dataset,reports]"], cwd=ROOT)

    python = _python(venv) if venv.exists() else Path(sys.executable)
    checks = [
        [str(python), "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", ".pytest-release-tmp"],
        [str(python), "-m", "nyssa_bench.cli", "list-suites"],
        [str(python), "scripts/validate_configs.py"],
        [str(python), "scripts/release_checklist.py"],
    ]
    for command in checks:
        _run(command, cwd=ROOT)

    print("release check passed")
    return 0


def _python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _run(command: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=cwd or ROOT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
