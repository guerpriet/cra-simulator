"""Launcher: starts `streamlit run crasim2_simple/app.py`."""
from __future__ import annotations
import os, sys
from pathlib import Path

def main() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path: sys.path.insert(0, str(here))
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", str(here / "app.py"),
                "--server.headless=false",
                "--browser.gatherUsageStats=false",
                *sys.argv[1:]]
    os.environ.setdefault("PYTHONPATH", str(here))
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
