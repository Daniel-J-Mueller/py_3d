import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from py_3d.cli import main, scaffold_example_game, scaffold_starter, select_environment_options, write_prefab_docs


def test_help_lists_prefab_docs_option(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "--write-prefab-docs" in output
    assert "init" in output
    assert "game" in output


def test_write_prefab_docs(tmp_path):
    environment_path, objects_path = write_prefab_docs(tmp_path)

    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    objects = json.loads(objects_path.read_text(encoding="utf-8"))

    assert "fields" in environment
    assert "background" in environment["fields"]
    assert "box" in objects["supported_object_types"]


def test_select_environment_options_accepts_numbers_and_defaults():
    answers = iter(["2", "", "front", "3", "1", "soft"])

    selections = select_environment_options(input_func=lambda prompt: next(answers), output_func=lambda message: None)

    assert selections["canvas"] == "wide"
    assert selections["background"] == "studio_gray"
    assert selections["camera"] == "front"
    assert selections["lighting"] == "dramatic"
    assert selections["quality"] == "draft"
    assert selections["shadows"] == "soft"


def test_scaffold_starter_writes_runnable_scene(tmp_path):
    target = tmp_path / "starter"

    paths = scaffold_starter(target, force=True)
    objects = json.loads(paths["objects"].read_text(encoding="utf-8"))

    assert len(objects["objects"]) == 2
    runpy.run_path(str(paths["main"]), run_name="__main__")
    assert (target / "starter_render.png").exists()


def test_scaffold_example_game_writes_preview_runner(tmp_path):
    target = tmp_path / "USER-GAMES" / "example-game"

    paths = scaffold_example_game(target, force=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run([sys.executable, str(paths["runner"]), "--render-preview"], cwd=target, env=env)

    assert result.returncode == 0
    assert (target / "output" / "preview.png").exists()
