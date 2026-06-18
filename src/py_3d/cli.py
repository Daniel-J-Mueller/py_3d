"""Command line starter tools for py3dengine."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable


VERSION = "0.0.2"
DEFAULT_STARTER_DIR = "py3dengine_starter"


ENVIRONMENT_OPTION_DOC: dict[str, Any] = {
    "schema": "py3dengine.environment-options.v1",
    "description": "Editable prefab documentation for quickly choosing starter environment settings.",
    "usage": [
        "Run `py3dengine init` for an interactive walkthrough.",
        "Run `py3dengine init --defaults` to write the default starter scene without prompts.",
        "Copy a key from each options object into environment.json:selected_options to reconfigure manually.",
    ],
    "fields": {
        "canvas": {
            "prompt": "Render canvas",
            "default": "wide",
            "options": {
                "small": {
                    "label": "Small preview",
                    "description": "Fast 640 by 480 preview renders.",
                    "values": {"render_settings": {"width": 640, "height": 480}},
                },
                "wide": {
                    "label": "Wide preview",
                    "description": "A 960 by 540 starter frame for screenshots.",
                    "values": {"render_settings": {"width": 960, "height": 540}},
                },
                "square": {
                    "label": "Square preview",
                    "description": "A centered 720 by 720 composition.",
                    "values": {"render_settings": {"width": 720, "height": 720}},
                },
            },
        },
        "background": {
            "prompt": "Background color",
            "default": "studio_gray",
            "options": {
                "studio_gray": {
                    "label": "Studio gray",
                    "description": "Neutral gray background for inspecting shapes.",
                    "values": {"render_settings": {"background": [34, 34, 34]}},
                },
                "night": {
                    "label": "Night",
                    "description": "Dark background for emissive or high contrast scenes.",
                    "values": {"render_settings": {"background": [8, 9, 11]}},
                },
                "sky_blue": {
                    "label": "Sky blue",
                    "description": "Clear outdoor-style background.",
                    "values": {"render_settings": {"background": [120, 158, 190]}},
                },
                "warm_light": {
                    "label": "Warm light",
                    "description": "Soft warm backdrop for product-like previews.",
                    "values": {"render_settings": {"background": [188, 176, 154]}},
                },
            },
        },
        "camera": {
            "prompt": "Camera angle",
            "default": "isometric",
            "options": {
                "front": {
                    "label": "Front",
                    "description": "Straight-on view for layout checks.",
                    "values": {
                        "camera": {
                            "position": [0.0, 1.1, -5.8],
                            "target": [0.0, 0.5, 0.0],
                            "fov_degrees": 58.0,
                            "near": 0.1,
                            "far": 100.0,
                        }
                    },
                },
                "isometric": {
                    "label": "Isometric",
                    "description": "Three-quarter view that shows cube depth clearly.",
                    "values": {
                        "camera": {
                            "position": [3.2, 2.2, -5.2],
                            "target": [0.0, 0.45, 0.0],
                            "fov_degrees": 55.0,
                            "near": 0.1,
                            "far": 100.0,
                        }
                    },
                },
                "top": {
                    "label": "Top",
                    "description": "Overhead planning view.",
                    "values": {
                        "camera": {
                            "position": [0.0, 6.0, -0.1],
                            "target": [0.0, 0.0, 0.0],
                            "fov_degrees": 50.0,
                            "near": 0.1,
                            "far": 100.0,
                        }
                    },
                },
            },
        },
        "lighting": {
            "prompt": "Lighting preset",
            "default": "soft_studio",
            "options": {
                "soft_studio": {
                    "label": "Soft studio",
                    "description": "One sun and one nearby fill lamp.",
                    "values": {
                        "render_settings": {"ambient": 0.12, "bounce_light": 0.18, "light_wrap": 0.08},
                        "lights": [
                            {"type": "sun", "direction": [-0.35, -0.78, -0.45], "color": [255, 255, 245], "intensity": 0.95},
                            {"type": "lamp", "position": [2.5, 3.0, -2.0], "color": [255, 236, 204], "intensity": 1.7},
                        ],
                    },
                },
                "sunny": {
                    "label": "Sunny",
                    "description": "Bright directional light with light ambient fill.",
                    "values": {
                        "render_settings": {"ambient": 0.08, "bounce_light": 0.22, "light_wrap": 0.04},
                        "lights": [
                            {"type": "sun", "direction": [-0.55, -0.82, -0.28], "color": [255, 248, 224], "intensity": 1.25}
                        ],
                    },
                },
                "dramatic": {
                    "label": "Dramatic",
                    "description": "Low ambient light with a stronger side lamp.",
                    "values": {
                        "render_settings": {"ambient": 0.03, "bounce_light": 0.05, "light_wrap": 0.0},
                        "lights": [
                            {"type": "lamp", "position": [-2.6, 2.4, -3.2], "color": [255, 244, 224], "intensity": 3.0}
                        ],
                    },
                },
                "ambient_only": {
                    "label": "Ambient only",
                    "description": "Flat lighting for checking silhouettes and color.",
                    "values": {"render_settings": {"ambient": 0.65, "bounce_light": 0.0}, "lights": []},
                },
            },
        },
        "quality": {
            "prompt": "Render quality",
            "default": "balanced",
            "options": {
                "draft": {
                    "label": "Draft",
                    "description": "Quick settings for iteration.",
                    "values": {
                        "render_settings": {
                            "smooth_shading": False,
                            "tone_mapping": False,
                            "sphere_segments": 10,
                            "sphere_rings": 5,
                            "texture_size": 128,
                        }
                    },
                },
                "balanced": {
                    "label": "Balanced",
                    "description": "Good starter defaults for still images.",
                    "values": {
                        "render_settings": {
                            "smooth_shading": True,
                            "tone_mapping": True,
                            "sphere_segments": 16,
                            "sphere_rings": 8,
                            "texture_size": 256,
                        }
                    },
                },
                "smooth": {
                    "label": "Smooth",
                    "description": "Higher geometry settings for nicer generated primitives.",
                    "values": {
                        "render_settings": {
                            "smooth_shading": True,
                            "tone_mapping": True,
                            "sphere_segments": 28,
                            "sphere_rings": 14,
                            "texture_size": 384,
                        }
                    },
                },
            },
        },
        "shadows": {
            "prompt": "Shadow style",
            "default": "off",
            "options": {
                "off": {
                    "label": "Off",
                    "description": "Fastest option for the starter cubes.",
                    "values": {"render_settings": {"ray_traced_shadows": False, "shadow_samples": 1, "shadow_softness": 0.0}},
                },
                "hard": {
                    "label": "Hard",
                    "description": "Single-sample ray-traced shadows.",
                    "values": {"render_settings": {"ray_traced_shadows": True, "shadow_samples": 1, "shadow_softness": 0.0}},
                },
                "soft": {
                    "label": "Soft",
                    "description": "Multi-sample shadows with a little spread.",
                    "values": {"render_settings": {"ray_traced_shadows": True, "shadow_samples": 5, "shadow_softness": 0.55}},
                },
            },
        },
    },
}


OBJECT_OPTION_DOC: dict[str, Any] = {
    "schema": "py3dengine.object-options.v1",
    "description": "Prefab object docs for the generated two-cube starter scene.",
    "supported_object_types": {
        "box": {
            "required": ["type", "name", "center", "size", "material"],
            "properties": {
                "center": "Three numbers for the cube center in world space.",
                "size": "Three positive numbers for width, height, and depth.",
                "material.color": "RGB array such as [220, 220, 220].",
                "material.roughness": "0.0 to 1.0 diffuse roughness.",
                "material.specular": "0.0 to 1.0 highlight strength.",
                "material.reflectivity": "0.0 to 1.0 reflection contribution.",
            },
        }
    },
    "starter_prefabs": {
        "two_cubes": {
            "description": "Two boxes with basic material properties, ready to render.",
            "object_count": 2,
        }
    },
}


def default_environment_selections() -> dict[str, str]:
    """Return the default option key for every documented environment field."""

    return {
        key: str(field["default"])
        for key, field in ENVIRONMENT_OPTION_DOC["fields"].items()
    }


def environment_from_selections(selections: dict[str, str] | None = None, *, name: str = "starter_cubes") -> dict[str, Any]:
    """Build an environment JSON payload from documented option keys."""

    selected = default_environment_selections()
    if selections:
        selected.update(selections)
    _validate_selections(selected)

    environment: dict[str, Any] = {
        "schema": "py3dengine.environment.v1",
        "name": name,
        "selected_options": selected,
        "docs": "environment-options.json",
        "output": "starter_render.png",
        "camera": {},
        "render_settings": {},
        "lights": [],
    }
    for field_name, field in ENVIRONMENT_OPTION_DOC["fields"].items():
        option = field["options"][selected[field_name]]
        _deep_update(environment, option.get("values", {}))
    return environment


def two_cube_objects() -> dict[str, Any]:
    """Return the default two-cube object JSON payload."""

    return {
        "schema": "py3dengine.objects.v1",
        "docs": "objects-options.json",
        "objects": [
            {
                "type": "box",
                "name": "left_cube",
                "center": [-0.7, 0.0, 0.0],
                "size": [1.0, 1.0, 1.0],
                "material": {
                    "color": [222, 222, 222],
                    "roughness": 0.35,
                    "specular": 0.18,
                    "shininess": 28.0,
                    "reflectivity": 0.02,
                },
            },
            {
                "type": "box",
                "name": "right_cube",
                "center": [0.78, 0.15, 0.32],
                "size": [0.86, 1.3, 0.86],
                "material": {
                    "color": [156, 156, 156],
                    "roughness": 0.48,
                    "specular": 0.12,
                    "shininess": 20.0,
                    "reflectivity": 0.0,
                },
            },
        ],
    }


def write_prefab_docs(output_dir: str | Path, *, force: bool = False) -> tuple[Path, Path]:
    """Write editable JSON docs for environment and object options."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    environment_path = target / "environment-options.json"
    objects_path = target / "objects-options.json"
    _write_json(environment_path, ENVIRONMENT_OPTION_DOC, force=force)
    _write_json(objects_path, OBJECT_OPTION_DOC, force=force)
    return environment_path, objects_path


def scaffold_starter(
    output_dir: str | Path,
    *,
    name: str = "starter_cubes",
    selections: dict[str, str] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Write docs, JSON config, and a starter Python render file."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    environment = environment_from_selections(selections, name=name)
    objects = two_cube_objects()

    docs_paths = write_prefab_docs(target, force=force)
    environment_path = target / "environment.json"
    objects_path = target / "objects.json"
    main_path = target / "main.py"
    _write_json(environment_path, environment, force=force)
    _write_json(objects_path, objects, force=force)
    _write_text(main_path, starter_python_source(), force=force)
    return {
        "environment_docs": docs_paths[0],
        "objects_docs": docs_paths[1],
        "environment": environment_path,
        "objects": objects_path,
        "main": main_path,
    }


def starter_python_source() -> str:
    """Return the generated starter Python source."""

    return dedent(
        '''\
        """Starter py3dengine scene generated by `py3dengine init`."""

        from __future__ import annotations

        import json
        from pathlib import Path

        from py_3d import Box, Camera, Lamp, Material, RenderEngine, RenderSettings, Scene, Sun


        ROOT = Path(__file__).resolve().parent


        def read_json(path: Path) -> dict:
            return json.loads(path.read_text(encoding="utf-8"))


        def material_from_json(config: dict) -> Material:
            return Material(
                color=config.get("color", [255, 255, 255]),
                roughness=config.get("roughness", 0.0),
                specular=config.get("specular", 0.0),
                shininess=config.get("shininess", 32.0),
                reflectivity=config.get("reflectivity", 0.0),
            )


        def object_from_json(config: dict):
            kind = config.get("type")
            if kind == "box":
                return Box(
                    center=config["center"],
                    size=config["size"],
                    material=material_from_json(config.get("material", {})),
                )
            raise ValueError(f"Unsupported object type: {kind}")


        def light_from_json(config: dict):
            kind = config.get("type")
            if kind == "sun":
                return Sun(
                    direction=config["direction"],
                    color=config.get("color", [255, 255, 255]),
                    intensity=config.get("intensity", 1.0),
                )
            if kind == "lamp":
                return Lamp(
                    position=config["position"],
                    color=config.get("color", [255, 255, 255]),
                    intensity=config.get("intensity", 1.0),
                    radius=config.get("radius"),
                )
            raise ValueError(f"Unsupported light type: {kind}")


        def build_scene(environment: dict, objects_doc: dict) -> tuple[Scene, Camera, RenderSettings]:
            settings = RenderSettings(**environment.get("render_settings", {}))
            camera_config = environment["camera"]
            camera = Camera(
                position=camera_config["position"],
                target=camera_config["target"],
                fov_degrees=camera_config.get("fov_degrees", 60.0),
                near=camera_config.get("near", 0.1),
                far=camera_config.get("far", 1000.0),
            )
            scene = Scene(background=settings.background)
            scene.add(*(object_from_json(item) for item in objects_doc.get("objects", [])))
            scene.add_light(*(light_from_json(item) for item in environment.get("lights", [])))
            return scene, camera, settings


        def main() -> None:
            environment = read_json(ROOT / "environment.json")
            objects_doc = read_json(ROOT / "objects.json")
            scene, camera, settings = build_scene(environment, objects_doc)
            output = ROOT / environment.get("output", "starter_render.png")
            RenderEngine().render(scene, camera, settings).to_png(output)
            print(f"Wrote {output}")


        if __name__ == "__main__":
            main()
        '''
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="py3dengine",
        description="Create starter py3dengine JSON scenes and prefab option docs.",
        epilog=dedent(
            """\
            Examples:
              py3dengine --write-prefab-docs USER/prefab-docs
              py3dengine docs --output USER/prefab-docs
              py3dengine init --output USER/projects/starter-cubes
              py3dengine init --defaults --output starter-cubes
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"py3dengine {VERSION}")
    parser.add_argument(
        "--write-prefab-docs",
        metavar="DIR",
        help="Pull editable prefab JSON docs for environment settings and two-cube objects into DIR.",
    )
    parser.add_argument(
        "--replace-prefab-docs",
        action="store_true",
        help="Overwrite files when used with --write-prefab-docs.",
    )

    subparsers = parser.add_subparsers(dest="command")
    docs = subparsers.add_parser("docs", help="Write prefab JSON option docs.")
    docs.add_argument("--output", "-o", default="py3dengine-prefab-docs", help="Directory to receive JSON docs.")
    docs.add_argument("--force", action="store_true", help="Overwrite existing doc files.")

    init = subparsers.add_parser("init", help="Walk through options and write a starter JSON scene.")
    init.add_argument("--output", "-o", default=DEFAULT_STARTER_DIR, help="Directory to receive starter files.")
    init.add_argument("--name", default="starter_cubes", help="Scene name written into environment.json.")
    init.add_argument("--defaults", action="store_true", help="Use documented default options without prompting.")
    init.add_argument("--force", action="store_true", help="Overwrite existing generated files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.write_prefab_docs:
        paths = write_prefab_docs(args.write_prefab_docs, force=args.replace_prefab_docs)
        print(f"Wrote {paths[0]}")
        print(f"Wrote {paths[1]}")
        return 0

    if args.command == "docs":
        paths = write_prefab_docs(args.output, force=args.force)
        print(f"Wrote {paths[0]}")
        print(f"Wrote {paths[1]}")
        return 0

    if args.command == "init":
        selections = default_environment_selections() if args.defaults else select_environment_options()
        paths = scaffold_starter(args.output, name=args.name, selections=selections, force=args.force)
        print(f"Wrote {paths['environment_docs']}")
        print(f"Wrote {paths['objects_docs']}")
        print(f"Wrote {paths['environment']}")
        print(f"Wrote {paths['objects']}")
        print(f"Wrote {paths['main']}")
        print(f"Run: python {paths['main']}")
        return 0

    parser.print_help()
    return 0


def select_environment_options(
    *,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> dict[str, str]:
    """Interactively choose one documented option for every environment field."""

    selections: dict[str, str] = {}
    fields = ENVIRONMENT_OPTION_DOC["fields"]
    for field_name, field in fields.items():
        default = str(field["default"])
        options = field["options"]
        option_keys = tuple(options.keys())
        output_func("")
        output_func(f"{field['prompt']}:")
        for index, key in enumerate(option_keys, start=1):
            option = options[key]
            output_func(f"  {index}. {key} - {option['label']}: {option['description']}")
        while True:
            try:
                raw = input_func(f"Select {field_name} [{default}]: ").strip()
            except EOFError:
                raw = ""
            if not raw:
                selections[field_name] = default
                break
            if raw.isdigit():
                index = int(raw)
                if 1 <= index <= len(option_keys):
                    selections[field_name] = option_keys[index - 1]
                    break
            if raw in options:
                selections[field_name] = raw
                break
            output_func(f"Choose one of: {', '.join(option_keys)}")
    return selections


def _validate_selections(selections: dict[str, str]) -> None:
    fields = ENVIRONMENT_OPTION_DOC["fields"]
    for field_name, field in fields.items():
        selected = selections.get(field_name)
        if selected not in field["options"]:
            choices = ", ".join(field["options"].keys())
            raise ValueError(f"{field_name} must be one of: {choices}")


def _deep_update(target: dict[str, Any], values: dict[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    _write_text(path, json.dumps(payload, indent=2) + "\n", force=force)


def _write_text(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite it")
    path.write_text(content, encoding="utf-8")


__all__ = [
    "ENVIRONMENT_OPTION_DOC",
    "OBJECT_OPTION_DOC",
    "default_environment_selections",
    "environment_from_selections",
    "scaffold_starter",
    "select_environment_options",
    "starter_python_source",
    "two_cube_objects",
    "write_prefab_docs",
]


if __name__ == "__main__":
    raise SystemExit(main())
