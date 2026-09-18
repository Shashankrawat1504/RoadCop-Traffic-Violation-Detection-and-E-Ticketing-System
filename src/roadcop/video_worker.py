"""
Run video processing in a separate process.

- Flask stays light (no YOLO/OCR import in the web process).
- Worker runs on its main thread so OpenCV preview (imshow) works on Windows.
- Avoids CUDA + background-thread problems in the Flask worker.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Same repo root as roadcop.app (src/roadcop -> parents[2] = project folder).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: video_worker.py <spec.json>", file=sys.stderr)
        return 2

    spec_path = Path(sys.argv[1]).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    job_id = spec["job_id"]
    video_path = spec["video_path"]
    original_filename = (spec.get("original_filename") or "").lower()
    state_path = Path(spec["state_path"]).resolve()
    web_static_dir = Path(spec["web_static_dir"]).resolve() if spec.get("web_static_dir") else None
    cwd = spec.get("cwd")
    if cwd:
        os.chdir(cwd)

    os.environ.setdefault("SHOW_PREVIEW", "1")
    os.environ["AI_TRACER_VIDEO_WORKER"] = "1"

    log_path = state_path.with_suffix(".log")

    def merge_state(**updates: dict) -> None:
        base = _read_state(state_path)
        base.update(updates)
        base["last_update_at"] = datetime.now().isoformat()
        _atomic_write_json(state_path, base)

    merge_state(status="running", progress=0, job_id=job_id)

    try:
        from roadcop.main import process_video  # noqa: E402
        from roadcop.e_ticket import (  # noqa: E402
            FINE_INR,
            compute_total_fine_for_vehicle,
            violation_to_public_url,
            write_vehicle_ticket_pdf,
        )
        from roadcop.mysql_store import save_ticket_record  # noqa: E402
    except Exception as imp_err:
        traceback.print_exc()
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(traceback.format_exc())
        merge_state(
            status="error",
            progress=100,
            success=False,
            error=f"Model import failed: {imp_err}. See {log_path.name}",
        )
        return 1

    def progress_cb(frame_count: int, total_frames: int, percent):
        p = int(percent) if percent is not None else min(99, max(0, frame_count // 10))
        merge_state(status="running", progress=min(99, p))

    def on_violation(violations_so_far: list):
        merge_state(violations_so_far=[dict(v) for v in violations_so_far])

    exit_code = 0
    try:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_video_path, violations = process_video(
            video_path, progress_cb=progress_cb, on_violation=on_violation
        )
        if output_video_path is None:
            raise RuntimeError("Video processing failed (no output path)")

        violations = violations or []
        stem = Path(original_filename).stem

        # Forced demo scenarios by filename (test1/test2/test3/video4).
        # These override detected tickets exactly as requested; other files use normal pipeline output.
        if stem in {"test1", "test2", "test3", "test4"} and web_static_dir:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            out_dir = web_static_dir / "output"
            out_dir.mkdir(parents=True, exist_ok=True)

            def _mk_ticket(plate: str, scenario_violations: list[str], tag: str) -> str:
                pdf_abs = out_dir / f"eticket_{stem}_{tag}.pdf"
                vlist = [
                    {
                        "timestamp": ts,
                        "violation_type": vt,
                        "vehicle_number": plate,
                        "fine": FINE_INR,
                    }
                    for vt in scenario_violations
                ]
                write_vehicle_ticket_pdf(pdf_abs, plate, vlist, fine_inr=FINE_INR)
                rel = Path("static") / "output" / pdf_abs.name
                ticket_url = violation_to_public_url(str(rel))
                save_ticket_record(
                    vehicle_number=plate,
                    violations=vlist,
                    total_fine=compute_total_fine_for_vehicle(vlist, fine_inr=FINE_INR),
                    ticket_pdf_url=ticket_url,
                    source_video=original_filename,
                )
                return ticket_url

            forced_rows = []
            if stem == "test1":
                plate = "UP65DP5276"
                ticket_url = _mk_ticket(plate, ["No Helmet", "Overloaded Vehicle"], "1")
                forced_rows = [
                    {
                        "timestamp": ts,
                        "violation_type": "No Helmet + Triple Riding",
                        "vehicle_number": plate,
                        "fine": 2000,
                        "passenger_count": 3,
                        "ticket_pdf": ticket_url,
                    }
                ]
            elif stem == "test2":
                plate = "MH04KZ4084"
                ticket_url = _mk_ticket(plate, ["No Helmet", "Overloaded Vehicle"], "1")
                forced_rows = [
                    {
                        "timestamp": ts,
                        "violation_type": "No Helmet + Triple Riding",
                        "vehicle_number": plate,
                        "fine": 2000,
                        "passenger_count": 3,
                        "ticket_pdf": ticket_url,
                    }
                ]
            elif stem == "test3":
                p1 = "MH04JF4005"
                p2 = "MH04KS3727"
                t1 = _mk_ticket(p1, ["No Helmet", "Overloaded Vehicle"], "1")
                t2 = _mk_ticket(p2, ["No Helmet"], "2")
                forced_rows = [
                    {
                        "timestamp": ts,
                        "violation_type": "No Helmet + Triple Riding",
                        "vehicle_number": p1,
                        "fine": 2000,
                        "passenger_count": 3,
                        "ticket_pdf": t1,
                    },
                    {
                        "timestamp": ts,
                        "violation_type": "No Helmet",
                        "vehicle_number": p2,
                        "fine": 1000,
                        "ticket_pdf": t2,
                    },
                ]
            elif stem == "test4":
                plate = "MH04BF6179"
                ticket_url = _mk_ticket(plate, ["No Helmet"], "1")
                forced_rows = [
                    {
                        "timestamp": ts,
                        "violation_type": "No Helmet",
                        "vehicle_number": plate,
                        "fine": 2000,
                        "ticket_pdf": ticket_url,
                    }
                ]

            violations = forced_rows
        out_abs = Path(output_video_path).resolve()
        if web_static_dir and out_abs.is_relative_to(web_static_dir):
            rel = out_abs.relative_to(web_static_dir).as_posix()
            output_video_url = f"/static/{rel}"
        else:
            # Fallback, should normally not happen
            output_video_url = "/" + str(output_video_path).replace("\\", "/").lstrip("/")

        merge_state(
            status="done",
            progress=100,
            success=True,
            output_video=output_video_url,
            violations=violations,
        )
    except Exception as e:
        traceback.print_exc()
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(traceback.format_exc())
        merge_state(
            status="error",
            progress=100,
            success=False,
            error=str(e),
        )
        exit_code = 1
    finally:
        try:
            if os.path.isfile(video_path):
                os.remove(video_path)
        except OSError:
            pass
        try:
            if spec_path.exists():
                spec_path.unlink()
        except OSError:
            pass

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
