import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "USER" / "demos"


def test_live_menu_settings_inventory_covers_showcase_demos():
    settings = json.loads((ROOT / "USER" / "settings.json").read_text(encoding="utf-8"))
    live_settings = settings["live_menu_settings"]

    for name in (
        "global",
        "fruit_bowl",
        "capsule_walk",
        "fan_cloth_water",
        "wind_pool_water",
        "sea_lion_asset",
        "procedural_hills",
    ):
        assert name in live_settings


def test_procedural_demo_stays_under_user_demos():
    source = DEMO_DIR / "19_live_procedural_environment.py"

    assert source.exists()
    assert not (ROOT / "examples" / "procedural_environment_demo.py").exists()

    if str(DEMO_DIR) not in sys.path:
        sys.path.insert(0, str(DEMO_DIR))
    spec = importlib.util.spec_from_file_location("procedural_environment_demo_layout", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert "USER\\environments" not in str(module.OUTPUT_DIR)
    assert "USER/environments" not in str(module.OUTPUT_DIR)
