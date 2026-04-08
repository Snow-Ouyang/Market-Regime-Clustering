from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    runpy.run_path(str(ROOT / "reporting" / "run_stress_aware_extension_interpretation.py"), run_name="__main__")
    runpy.run_path(str(ROOT / "reporting" / "build_final_outputs.py"), run_name="__main__")


if __name__ == "__main__":
    main()
