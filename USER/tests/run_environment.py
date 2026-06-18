"""Run or list saved USER environments."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
USER_DIR = ROOT / "USER"
ENV_DIR = USER_DIR / "environments"
SETTINGS_PATH = USER_DIR / "settings.json"


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    return {}


def environment_path(name: str) -> Path:
    return ENV_DIR / name / "environment.json"


def load_environment(name: str) -> dict:
    path = environment_path(name)
    if not path.exists():
        available = ", ".join(environment_names())
        raise SystemExit(f"Unknown environment {name!r}. Available: {available}")
    environment = json.loads(path.read_text(encoding="utf-8"))
    environment.setdefault("environment_dir", str(path.parent.relative_to(ROOT)))
    environment.setdefault("rendering_dir", str(path.parent.joinpath("renderings").relative_to(ROOT)))
    environment.setdefault("baking_dir", str(path.parent.joinpath("baking").relative_to(ROOT)))
    environment.setdefault("render_data", str(path.parent.joinpath("render-data.json").relative_to(ROOT)))
    return environment


def environment_names() -> list[str]:
    names = []
    for path in (ENV_DIR.iterdir() if ENV_DIR.exists() else ()):
        if path.is_dir() and path.joinpath("environment.json").exists():
            names.append(path.name)
    return sorted(names)


def selected_renders(environment: dict, only: set[str], skip: set[str]) -> list[dict]:
    renders = list(environment.get("renders", []))
    if only:
        renders = [render for render in renders if render.get("name") in only]
    if skip:
        renders = [render for render in renders if render.get("name") not in skip]
    return renders


def ensure_environment_dirs(environment: dict) -> None:
    for key in ("rendering_dir", "baking_dir"):
        target = ROOT / environment[key]
        target.mkdir(parents=True, exist_ok=True)


def run_command(command: list[str], *, dry_run: bool = False) -> dict:
    started = time.perf_counter()
    if dry_run:
        return {"command": command, "seconds": 0.0, "returncode": 0, "dry_run": True}
    result = subprocess.run(command, cwd=ROOT)
    elapsed = time.perf_counter() - started
    return {"command": command, "seconds": elapsed, "returncode": result.returncode, "dry_run": False}


def system_specs() -> dict:
    gpu = _nvidia_smi()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu or "not detected",
    }


def print_specs(specs: dict) -> None:
    print("system specs:")
    print(f"  python: {specs['python']}")
    print(f"  platform: {specs['platform']}")
    print(f"  cpu: {specs['cpu']} ({specs['cpu_count']} logical)")
    print(f"  gpu: {specs['gpu']}")


def _nvidia_smi() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return "; ".join(lines) if lines else None


def write_render_data(environment: dict, records: list[dict], specs: dict) -> None:
    target = ROOT / environment["render_data"]
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = []
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8")).get("runs", [])
    payload = {
        "environment": environment["name"],
        "rendering_dir": environment["rendering_dir"],
        "baking_dir": environment["baking_dir"],
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_specs": specs,
        "runs": previous + records,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_bake_manifest(environment: dict, settings: dict, specs: dict) -> None:
    target = ROOT / environment["baking_dir"] / "render-profile.json"
    payload = {
        "environment": environment["name"],
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rendering_dir": environment["rendering_dir"],
        "baking_dir": environment["baking_dir"],
        "standard_render_specs": settings.get("standard_render_specs", {}),
        "system_specs": specs,
        "status": "metadata-only",
        "notes": [
            "This file is the first persistent baking contract for the environment.",
            "Future passes should add prepared geometry, lightmaps, texture atlases, or static render caches here.",
        ],
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run saved USER environment renders or live demos.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--settings", action="store_true", help="Print USER/settings.json and exit.")
    parser.add_argument("--environment", "-e")
    parser.add_argument("--variant", action="append", default=[], help="Render variant name to run. Can be repeated.")
    parser.add_argument("--skip", action="append", default=[], help="Render variant name to skip. Can be repeated.")
    parser.add_argument("--live", nargs="?", const="default", help="Launch a live command by name.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--specs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.settings:
        print(json.dumps(load_settings(), indent=2))
        return
    if args.list:
        for name in environment_names():
            environment = load_environment(name)
            print(f"{name}: {environment.get('description', '')}")
        return
    if not args.environment:
        raise SystemExit("Pass --environment NAME, --settings, or --list.")

    environment = load_environment(args.environment)
    ensure_environment_dirs(environment)
    specs = system_specs()
    settings = load_settings()
    write_bake_manifest(environment, settings, specs)
    if args.specs:
        print_specs(specs)
    if args.live is not None:
        live_commands = environment.get("live", {})
        live_name = args.live
        if live_name not in live_commands and live_name == "default" and live_commands:
            live_name = next(iter(live_commands))
        command = live_commands.get(live_name)
        if command is None:
            available = ", ".join(live_commands)
            raise SystemExit(f"Unknown live command {args.live!r}. Available: {available}")
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT)
        return

    records: list[dict] = []
    only = set(args.variant)
    skip = set(args.skip)
    for render in selected_renders(environment, only, skip):
        command = render["command"]
        print(f"[{environment['name']}:{render['name']}] {' '.join(command)}")
        record = run_command(command, dry_run=args.dry_run)
        record["variant"] = render["name"]
        record["expected_seconds"] = render.get("seconds")
        record["outputs"] = render.get("outputs", [])
        records.append(record)
        if record["returncode"] != 0:
            write_render_data(environment, records, specs)
            raise SystemExit(record["returncode"])
    write_render_data(environment, records, specs)


if __name__ == "__main__":
    main()
