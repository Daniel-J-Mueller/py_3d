"""Live fruit bowl variant with standalone RGB blinking bulbs."""

from __future__ import annotations

import sys

from fruit_bowl_live import main


def _ensure_arg(flag: str, value: str) -> None:
    if flag not in sys.argv and not any(arg.startswith(f"{flag}=") for arg in sys.argv):
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _ensure_arg("--light-mode", "rgb-bulbs")
    _ensure_arg("--quality", "high")
    _ensure_arg("--label", "RGB BULB BOWL")
    main()
