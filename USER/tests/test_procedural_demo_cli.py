import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "USER" / "demos"
DEMO_PATH = DEMO_DIR / "19_live_procedural_environment.py"


def _load_demo():
    if str(DEMO_DIR) not in sys.path:
        sys.path.insert(0, str(DEMO_DIR))
    spec = importlib.util.spec_from_file_location("procedural_environment_demo_cli", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_procedural_demo_defaults_to_live_mode(monkeypatch):
    demo = _load_demo()
    monkeypatch.setattr(sys, "argv", [str(DEMO_PATH)])

    args = demo.parse_args()

    assert args.mode == "live"
    assert demo.should_run_live(args) is True
    assert args.live_backend == "auto"
    assert args.still_renderer == "auto"
    assert args.chunk_worker is True
    assert args.preload_margin >= 1
    assert args.chunk_worker_mode == "process"
    assert args.chunk_activation_rate >= 100
    assert args.reflection_bounces >= 1
    assert args.sky_cycle is True


def test_procedural_demo_still_mode_is_explicit_and_repo_anchored(monkeypatch):
    demo = _load_demo()
    monkeypatch.chdir(DEMO_DIR)
    monkeypatch.setattr(sys, "argv", [str(DEMO_PATH), "--still"])

    args = demo.parse_args()

    assert args.mode == "still"
    assert demo.should_run_live(args) is False
    assert args.output.is_absolute()
    assert ROOT in args.output.parents
