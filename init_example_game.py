"""Create the turnkey USER-GAMES/example-game starter project.

Run from the repository root:

    python init_example_game.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from py_3d.cli import DEFAULT_GAME_DIR, scaffold_example_game


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate USER-GAMES/example-game for new py3dengine projects.")
    parser.add_argument("--output", "-o", default=str(DEFAULT_GAME_DIR), help="Target game directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing generated files.")
    args = parser.parse_args()

    paths = scaffold_example_game(Path(args.output), force=args.force)
    for key in ("readme", "environment", "objects", "settings", "game", "runner"):
        print(f"Wrote {paths[key]}")
    print(f"Run: python {paths['runner']}")


if __name__ == "__main__":
    main()
