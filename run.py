from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from roadcop.app import app, init_db

if __name__ == "__main__":
    init_db()
    app.run(debug=True, use_reloader=False)
