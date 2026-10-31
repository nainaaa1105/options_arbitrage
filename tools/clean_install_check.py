"""Verify the project in a brand-new virtual environment: install pinned requirements, then run the offline suite.

This DOES download packages when run without --print-plan. It is therefore never invoked by the test suite;
run it deliberately (CI or a developer machine). The CPU PyTorch wheel needs its own index, as documented in
requirements/deep-hedging-lock.txt.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def venv_python(folder):
    return folder / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def build_commands(folder, requirement_files, dry_run=False):
    """Return the exact commands, in order, as argument lists (pure; nothing is executed here)."""
    py = str(venv_python(Path(folder)))
    commands = [[sys.executable, "-m", "venv", str(folder)], [py, "-m", "pip", "install", "--upgrade", "pip"]]
    for file in requirement_files:
        command = [py, "-m", "pip", "install", "--no-input"]
        if dry_run:
            command.append("--dry-run")
        commands.append(command + ["-r", str(file)])
    if not dry_run:
        commands += [[py, "-m", "pip", "check"], [py, str(ROOT / "tools/run_offline_tests.py")]]
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, nargs="+", default=[ROOT / "requirements.txt", ROOT / "requirements/deep-hedging-lock.txt"])
    parser.add_argument("--dry-run", action="store_true", help="Resolve and download metadata only; skip the test run")
    parser.add_argument("--print-plan", action="store_true", help="Print the commands without running anything")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="clean_install_") as scratch:
        commands = build_commands(Path(scratch) / "venv", args.requirements, args.dry_run)
        if args.print_plan:
            for command in commands:
                print(" ".join(command))
            return 0
        for command in commands:
            print("RUN", " ".join(command), flush=True)
            if subprocess.run(command, cwd=ROOT).returncode:
                print("CLEAN INSTALL CHECK FAILED at:", " ".join(command))
                return 1
    print("CLEAN INSTALL CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
