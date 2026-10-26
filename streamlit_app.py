"""
streamlit_app.py - Root entrypoint for Streamlit Community Cloud (share.streamlit.io).
Automatically forwards execution to src/app/dashboard.py.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute the complete SOC Dashboard
dashboard_path = ROOT_DIR / "src" / "app" / "dashboard.py"
with open(dashboard_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(code, globals())
