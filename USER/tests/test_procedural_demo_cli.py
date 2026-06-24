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
    assert args.max_pending_chunks >= 1
    assert args.chunk_worker_mode == "thread"
    assert args.hlod_radius >= 0
    assert args.virtual_detail_budget >= 0
    assert args.virtual_pixel_error >= 0.5
    assert args.tree_lod_distance_chunks >= 1
    assert 1 <= args.chunk_activation_rate <= 12
    assert args.reflection_bounces >= 1
    assert args.sky_cycle is True


def test_procedural_fast_quality_caps_saved_view_radius(monkeypatch):
    demo = _load_demo()
    monkeypatch.setattr(sys, "argv", [str(DEMO_PATH), "--quality", "fast"])

    args = demo.parse_args()

    assert args.active_radius == demo.QUALITY_PROFILES["fast"]["active_radius"]
    assert args.hlod_radius == demo.QUALITY_PROFILES["fast"]["hlod_radius"]
    assert args.virtual_detail_budget == demo.QUALITY_PROFILES["fast"]["virtual_detail_budget"]
    assert args.tree_lod_distance_chunks == demo.QUALITY_PROFILES["fast"]["tree_lod_distance_chunks"]


def test_procedural_demo_still_mode_is_explicit_and_repo_anchored(monkeypatch):
    demo = _load_demo()
    monkeypatch.chdir(DEMO_DIR)
    monkeypatch.setattr(sys, "argv", [str(DEMO_PATH), "--still"])

    args = demo.parse_args()

    assert args.mode == "still"
    assert demo.should_run_live(args) is False
    assert args.output.is_absolute()
    assert ROOT in args.output.parents
