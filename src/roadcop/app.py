import json
import os
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
WEB_DIR = PROJECT_ROOT / "web"
WEB_STATIC_DIR = WEB_DIR / "static"
WEB_TEMPLATES_DIR = WEB_DIR / "templates"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "access_portal.db"
JOB_STATE_DIR = DATA_DIR / "runtime" / "job_state"
JOB_QUEUE_DIR = DATA_DIR / "runtime" / "job_queue"

load_dotenv(PROJECT_ROOT / ".env")

app = Flask(
    __name__,
    template_folder=str(WEB_TEMPLATES_DIR),
    static_folder=str(WEB_STATIC_DIR),
    static_url_path="/static",
)
app.secret_key = "your_secret_key"

os.makedirs(WEB_STATIC_DIR, exist_ok=True)
os.makedirs(JOB_STATE_DIR, exist_ok=True)
os.makedirs(JOB_QUEUE_DIR, exist_ok=True)
os.makedirs(WEB_STATIC_DIR / "uploads", exist_ok=True)
os.makedirs(WEB_STATIC_DIR / "output", exist_ok=True)


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _job_state_path(job_id: str) -> Path:
    return JOB_STATE_DIR / f"{job_id}.json"


def _spawn_video_worker(spec_path: Path) -> None:
    """Separate process: loads YOLO/OCR on main thread, preview works."""
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.Popen(
        [sys.executable, "-m", "roadcop.video_worker", str(spec_path.resolve())],
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        env=env,
        close_fds=False,
    )


# ---- auth / routes ----


def check_credentials(name, user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE name=? AND user_id=?", (name, user_id))
    user = cursor.fetchone()
    conn.close()
    return user is not None


@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    name = data.get("name")
    user_id = data.get("user_id")

    if check_credentials(name, user_id):
        session["user"] = name
        return jsonify({"success": True, "message": "Authentication successful", "redirect": url_for("dashboard")})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401


@app.route("/dashboard")
def dashboard():
    if "user" in session:
        return render_template("dashboard.html", user=session["user"])
    return redirect(url_for("home"))


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))


@app.route("/upload", methods=["POST"])
def upload():
    if "user" not in session:
        return jsonify({"success": False, "error": "Please login first"}), 401

    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    if not video_file.filename.lower().endswith((".mp4", ".avi", ".mov")):
        return jsonify({"success": False, "error": "Invalid file type. Please upload a video file"}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"upload_{timestamp}_{video_file.filename}"
        temp_path = WEB_STATIC_DIR / "uploads" / temp_filename
        video_file.save(str(temp_path))

        job_id = uuid.uuid4().hex
        state_path = _job_state_path(job_id)
        _atomic_write_json(
            state_path,
            {"status": "queued", "progress": 0, "job_id": job_id},
        )

        spec_path = JOB_QUEUE_DIR / f"{job_id}.json"
        spec = {
            "job_id": job_id,
            "video_path": str(temp_path.resolve()),
            "original_filename": video_file.filename,
            "state_path": str(state_path.resolve()),
            "cwd": str(PROJECT_ROOT),
            "web_static_dir": str(WEB_STATIC_DIR.resolve()),
        }
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

        _spawn_video_worker(spec_path)

        return jsonify({"success": True, "job_id": job_id})

    except Exception as e:
        if "temp_path" in locals() and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/upload/status/<job_id>")
def upload_status(job_id):
    if "user" not in session:
        return jsonify({"success": False, "error": "Please login first"}), 401

    state_path = _job_state_path(job_id)
    if not state_path.exists():
        return jsonify({"success": False, "error": "Job not found"}), 404

    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return jsonify({"success": False, "error": "Invalid job state"}), 500

    status = data.get("status", "unknown")

    if status == "done":
        return jsonify(
            {
                "success": True,
                "status": "done",
                "progress": data.get("progress", 100),
                "output_video": data.get("output_video"),
                "violations": data.get("violations", []),
            }
        )

    if status == "error":
        return jsonify(
            {
                "success": False,
                "status": "error",
                "progress": data.get("progress", 100),
                "error": data.get("error", "Unknown error"),
            }
        )

    payload = {
        "success": True,
        "status": status,
        "progress": int(data.get("progress", 0)),
    }
    v = data.get("violations_so_far")
    if v:
        payload["violations_so_far"] = v
    return jsonify(payload)


@app.route("/download/<path:filename>")
def download(filename):
    if "user" not in session:
        return redirect(url_for("home"))

    try:
        static_root = WEB_STATIC_DIR.resolve()
        target = (static_root / filename).resolve()
        if static_root not in target.parents and target != static_root:
            raise PermissionError("Invalid path")
        if not target.is_file():
            raise FileNotFoundError("File not found")
        rel = target.relative_to(static_root)
        return send_from_directory(str(static_root), str(rel).replace("\\", "/"), as_attachment=True)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 404


def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            user_id TEXT NOT NULL UNIQUE
        )
    """
    )
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO users (name, user_id) VALUES (?, ?)", ("Admin", "admin123"))
        cursor.execute("INSERT INTO users (name, user_id) VALUES (?, ?)", ("User", "user123"))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    app.run(debug=True, use_reloader=False)
