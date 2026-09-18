import os
import threading
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import cvzone
import torch
from paddleocr import PaddleOCR

from roadcop.e_ticket import FINE_INR, compute_total_fine_for_vehicle, violation_to_public_url, write_vehicle_ticket_pdf
from roadcop.image_to_text import predict_number_plate
from roadcop.mysql_store import save_ticket_record

# Initialize YOLO Model and OCR globally
def _select_device():
    if os.environ.get("FORCE_CPU", "0").strip().lower() in ("1", "true", "yes"):
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = _select_device()


def _log_torch_device():
    print(f"[RoadCop] YOLO device: {device}")
    if os.environ.get("FORCE_CPU", "0").strip().lower() in ("1", "true", "yes"):
        print("[RoadCop] FORCE_CPU=1 — skipping GPU hints.")
        return
    if device.type == "cuda":
        try:
            print(f"[RoadCop] GPU: {torch.cuda.get_device_name(0)} | CUDA (PyTorch): {torch.version.cuda}")
        except Exception:
            pass
        return
    # CPU-only wheels report torch.version.cuda is None
    if torch.version.cuda is None:
        print(
            "[RoadCop] PyTorch is built without CUDA (common: pip installed the CPU wheel). "
            "YOLO will run on CPU until you install a CUDA build."
        )
        print(
            "[RoadCop] Install GPU PyTorch (pick cu124/cu128 to match your driver; see pytorch.org): "
            "pip uninstall torch torchvision -y && "
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        )
    elif not torch.cuda.is_available():
        print(
            "[RoadCop] PyTorch has CUDA support but torch.cuda.is_available() is False. "
            "Update GPU drivers, install CUDA toolkit if needed, or reboot after driver install."
        )


REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_STATIC_DIR = REPO_ROOT / "web" / "static"

_log_torch_device()

def get_latest_yolov5_best_pt() -> str:
    """
    The training weights are typically saved under:
      yolov5/runs/train/exp*/weights/best.pt
    Pick the most recently modified one so the script works out-of-the-box.
    """
    candidates = list(REPO_ROOT.glob("yolov5/runs/train/exp*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find 'yolov5/runs/train/exp*/weights/best.pt' under the project. "
            "Train YOLO or copy your best.pt into that path."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])

weights_path = get_latest_yolov5_best_pt()

# Work around a common checkpoint portability issue:
# some YOLOv5/torch checkpoints saved on Linux include `pathlib.PosixPath`,
# which cannot be instantiated directly on Windows during `torch.load`.
if os.name == "nt":
    import pathlib

    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[attr-defined]
    pathlib.PurePosixPath = pathlib.PureWindowsPath  # type: ignore[attr-defined]

# Use cached repo if already downloaded to avoid Windows file-lock issues
# (torch.hub with force_reload=True repeatedly tries to overwrite master.zip).
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
model.to(device)
model.eval()
model.conf = float(os.environ.get("YOLO_CONF", "0.3"))

# Smaller = faster (typical: 416–640). Larger = more accurate on small objects.
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))
# Run YOLO every N frames; reuse previous detections on skipped frames (faster, boxes lag slightly).
INFERENCE_STRIDE = max(1, int(os.environ.get("INFERENCE_STRIDE", "1")))
# OCR is expensive; sample plate OCR every N frames for history/backfill.
PLATE_OCR_SAMPLE_STRIDE = max(1, int(os.environ.get("PLATE_OCR_SAMPLE_STRIDE", "8")))
MAX_PLATE_HISTORY = max(20, int(os.environ.get("MAX_PLATE_HISTORY", "120")))
if INFERENCE_STRIDE > 1:
    print(
        f"[RoadCop] INFERENCE_STRIDE={INFERENCE_STRIDE} "
        "(YOLO not on every frame — faster; boxes can lag one frame)."
    )
print(f"[RoadCop] YOLO_IMGSZ={YOLO_IMGSZ} | Tip: CUDA PyTorch + SHOW_PREVIEW=0 speeds things up.")
# `enable_mkldnn=False` is a compatibility workaround for some Windows/Paddle builds.
# Disable document-specific preprocessing/unwarping/orientation steps for speed:
# number plate crops don't need them and they can be extremely slow.
ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

# Class names
classNames = ["Helmet", "No Helmet", "Number plate", "Passenger Count"]

# Link a no-helmet head box to a plate by nearest-centre distance (overlap with plate is rare).
PLATE_LINK_MAX_DIAG_FRAC = float(os.environ.get("PLATE_LINK_MAX_DIAG_FRAC", "0.5"))


def _frame_diagonal(w: int, h: int) -> float:
    return float((w * w + h * h) ** 0.5)


def _box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _dist(a, b) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _nearest_plate_for_box(rider_box, plates: list, max_dist: float):
    """Return (plate_box, distance) or (None, inf)."""
    if not plates:
        return None, float("inf")
    rcx, rcy = _box_center(rider_box)
    best = None
    best_d = float("inf")
    for pb in plates:
        pcx, pcy = _box_center(pb)
        d = _dist((rcx, rcy), (pcx, pcy))
        if d < best_d:
            best_d = d
            best = pb
    if best is not None and best_d <= max_dist:
        return best, best_d
    return None, best_d


def _count_riders_in_passenger_box(passenger_box, helmet_detections: list) -> int:
    """Count Helmet / No Helmet detections whose centre lies inside the passenger ROI."""
    px1, py1, px2, py2 = passenger_box
    n = 0
    for h_x1, h_y1, h_x2, h_y2, _h_cls in helmet_detections:
        cx, cy = _box_center((h_x1, h_y1, h_x2, h_y2))
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            n += 1
    return n


def _nearest_plate_number_for_boxes(candidate_boxes: list, number_plates: list, frame_img, ocr_engine):
    """Attach violation to nearest detected plate for a cluster/box."""
    if not candidate_boxes:
        return "UNREAD"
    # Use centre of first candidate box as anchor
    anchor = _box_center(candidate_boxes[0])
    if number_plates:
        best = None
        best_d = float("inf")
        for pb in number_plates:
            d = _dist(anchor, _box_center(pb))
            if d < best_d:
                best_d = d
                best = pb
        if best is not None:
            px1, py1, px2, py2 = best
            crop = frame_img[py1:py2, px1:px2]
            if crop.size > 0:
                plate, _ = predict_number_plate(crop, ocr_engine)
                if plate:
                    return plate
    return "UNREAD"


def _resolve_unread_plates(violations: list[dict], plate_history: list[dict]) -> None:
    """Backfill UNREAD vehicle numbers using nearby plate reads across frames."""
    if not violations or not plate_history:
        return
    for v in violations:
        if (v.get("vehicle_number") or "").upper() != "UNREAD":
            continue
        anchor = v.get("_anchor")
        if not anchor:
            continue
        ax, ay = anchor
        best_text = None
        best_rank = -1.0
        for p in plate_history:
            px, py = p["center"]
            d = _dist((ax, ay), (px, py))
            if d > 320:
                continue
            # Prefer high confidence and near anchor.
            rank = float(p.get("score", 0.0)) - (d / 1000.0)
            if rank > best_rank:
                best_rank = rank
                best_text = p.get("text")
        if best_text:
            v["vehicle_number"] = best_text


def _open_video_writer(path: str, fps: float, size: tuple[int, int]):
    """Try codecs that often work in browsers (H.264) then fall back to mp4v."""
    w, h = size
    for tag in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*tag)
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if out.isOpened():
            return out, tag
        out.release()
    raise RuntimeError("Could not open VideoWriter for any tried codec")


def _attach_consolidated_tickets(violations: list[dict], run_ts: str) -> None:
    """Generate one PDF per vehicle and attach same ticket URL to each row."""
    grouped: dict[str, list[dict]] = {}
    for v in violations:
        plate = (v.get("vehicle_number") or "UNREAD").strip()
        grouped.setdefault(plate, []).append(v)

    seq = 0
    for plate, vlist in grouped.items():
        seq += 1
        relative = Path("static") / "output" / f"eticket_{run_ts}_vehicle_{seq}.pdf"
        absolute = WEB_STATIC_DIR / "output" / f"eticket_{run_ts}_vehicle_{seq}.pdf"
        write_vehicle_ticket_pdf(absolute, plate, vlist, fine_inr=FINE_INR)
        ticket_url = violation_to_public_url(str(relative))
        total_fine = compute_total_fine_for_vehicle(vlist, fine_inr=FINE_INR)
        save_ticket_record(
            vehicle_number=plate,
            violations=vlist,
            total_fine=total_fine,
            ticket_pdf_url=ticket_url,
            source_video=run_ts,
        )
        for v in vlist:
            v["ticket_pdf"] = ticket_url
            v.setdefault("fine", FINE_INR)


def process_video(video_path, progress_cb=None, on_violation=None):
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Error opening video file")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        if fps_raw is None or fps_raw < 1:
            fps = 25.0
        else:
            fps = float(fps_raw)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        if frame_width < 2 or frame_height < 2:
            cap.release()
            raise Exception("Invalid video dimensions")
        
        # Create output directory
        os.makedirs(WEB_STATIC_DIR / "output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = str(WEB_STATIC_DIR / "output" / f"output_{timestamp}.mp4")
        output, _codec_used = _open_video_writer(output_video_path, fps, (frame_width, frame_height))
        
        # Initialize tracking variables
        violations = []
        no_helmet_keys = set()
        overload_keys = set()
        plate_history = []
        diag = _frame_diagonal(frame_width, frame_height)
        max_plate_link = PLATE_LINK_MAX_DIAG_FRAC * diag
        frame_count = 0
        # OpenCV windows only work reliably on the main thread (Flask uses a worker thread).
        _pv = os.environ.get("SHOW_PREVIEW", "1").strip().lower()
        wants_preview = _pv in ("1", "true", "yes")
        on_main = threading.current_thread() is threading.main_thread()
        in_worker = os.environ.get("AI_TRACER_VIDEO_WORKER", "0") == "1"
        # Subprocess worker always runs on its main thread — preview works there.
        enable_preview = wants_preview and (on_main or in_worker)
        preview_failed = not enable_preview
        if wants_preview and not enable_preview and not in_worker:
            print(
                "[RoadCop] Preview skipped (not main thread). "
                "Upload uses a separate worker process with preview enabled."
            )
        cached_results = None

        _last_report_frame = 0

        def _report_progress(force: bool = False):
            nonlocal _last_report_frame
            if not callable(progress_cb):
                return
            if not force and frame_count - _last_report_frame < 3 and frame_count > 0:
                return
            _last_report_frame = frame_count
            try:
                if total_frames > 0:
                    pct = min(99, int((frame_count / total_frames) * 100))
                else:
                    pct = min(99, max(1, frame_count // 10))
                progress_cb(frame_count=frame_count, total_frames=total_frames, percent=pct)
            except Exception:
                pass

        def _append_violation(rec: dict) -> None:
            violations.append(rec)
            if callable(on_violation):
                try:
                    on_violation(list(violations))
                except Exception:
                    pass

        while True:
            success, img = cap.read()
            if not success:
                break
                
            frame_count += 1

            run_yolo = (frame_count - 1) % INFERENCE_STRIDE == 0 or cached_results is None
            if run_yolo:
                with torch.inference_mode():
                    results = model(img, size=YOLO_IMGSZ)
                cached_results = results
            else:
                results = cached_results
            
            # Store detections
            number_plates = []
            passenger_boxes = []
            helmet_detections = []
            
            # Process each detection
            for det in results.xyxy[0]:  # xyxy format
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                w, h = x2 - x1, y2 - y1
                
                # Draw bounding box and label
                color = (0, 255, 0)  # Default green color
                if classNames[int(cls)] == "No Helmet":
                    color = (0, 0, 255)  # Red for no helmet
                elif classNames[int(cls)] == "Helmet":
                    color = (0, 255, 0)  # Green for helmet
                
                cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=color)
                cvzone.putTextRect(img, f"{classNames[int(cls)]} {conf:.2f}", (x1, y1 - 10), 
                                scale=1.5, offset=10, thickness=2)
                
                # Store detections based on class
                if classNames[int(cls)] == "Number plate" and conf >= 0.3:
                    number_plates.append((x1, y1, x2, y2))
                elif classNames[int(cls)] == "Passenger Count" and conf >= 0.3:
                    passenger_boxes.append((x1, y1, x2, y2))
                elif classNames[int(cls)] in ["Helmet", "No Helmet"] and conf >= 0.3:
                    helmet_detections.append((x1, y1, x2, y2, int(cls)))
            
            # No helmet: pair each "No Helmet" box with nearest plate (not plate–helmet bbox overlap).
            nh_boxes = [
                (h[0], h[1], h[2], h[3])
                for h in helmet_detections
                if classNames[h[4]] == "No Helmet"
            ]
            for h_x1, h_y1, h_x2, h_y2 in nh_boxes:
                nearest_plate, _d = _nearest_plate_for_box(
                    (h_x1, h_y1, h_x2, h_y2), number_plates, max_plate_link
                )
                vehicle_number = None
                if nearest_plate is not None:
                    px1, py1, px2, py2 = nearest_plate
                    cropped_np = img[py1:py2, px1:px2]
                    if cropped_np.size > 0:
                        vehicle_number, _conf = predict_number_plate(cropped_np, ocr)
                if not vehicle_number:
                    vehicle_number = "UNREAD"

                hcx, hcy = _box_center((h_x1, h_y1, h_x2, h_y2))
                if vehicle_number != "UNREAD":
                    dedupe_key = ("no_helmet", vehicle_number)
                else:
                    dedupe_key = ("no_helmet_unread", int(hcx // 320), int(hcy // 320))
                if dedupe_key in no_helmet_keys:
                    continue
                no_helmet_keys.add(dedupe_key)

                if nearest_plate:
                    npx1, npy1, _npx2, _npy2 = nearest_plate
                    cvzone.putTextRect(
                        img,
                        f"Plate: {vehicle_number}",
                        (npx1, npy1 - 30),
                        scale=1.2,
                        offset=10,
                        thickness=2,
                    )
                _append_violation({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'violation_type': 'No Helmet',
                    'vehicle_number': vehicle_number,
                    'fine': FINE_INR,
                    '_anchor': _box_center((h_x1, h_y1, h_x2, h_y2)),
                })
                cvzone.putTextRect(
                    img,
                    "NO HELMET!",
                    (h_x1, h_y1 - 10),
                    scale=1.5,
                    offset=10,
                    thickness=2,
                    colorR=(0, 0, 255),
                )

            # Triple riding / overload: count helmet-class heads inside each Passenger Count ROI
            for px1, py1, px2, py2 in passenger_boxes:
                passenger_count = _count_riders_in_passenger_box(
                    (px1, py1, px2, py2), helmet_detections
                )
                if passenger_count > 2:
                    # Prefer real vehicle number over N/A by linking this ROI to nearest plate
                    vehicle_number = _nearest_plate_number_for_boxes(
                        [(px1, py1, px2, py2)], number_plates, img, ocr
                    )
                    obox_key = ("overload", vehicle_number, px1 // 20, py1 // 20, px2 // 20, py2 // 20)
                    if obox_key in overload_keys:
                        continue
                    overload_keys.add(obox_key)
                    cvzone.putTextRect(
                        img,
                        f"Overloaded: {passenger_count} Passengers",
                        (px1, py1 - 30),
                        scale=1.5,
                        offset=10,
                        thickness=2,
                        colorR=(255, 0, 0),
                    )
                    _append_violation({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'violation_type': 'Overloaded Vehicle',
                        'vehicle_number': vehicle_number,
                        'passenger_count': passenger_count,
                        'fine': FINE_INR,
                        '_anchor': _box_center((px1, py1, px2, py2)),
                    })

            # Additional robust rule: many models output 2 passenger boxes for triple-riding (excluding rider).
            # So treat >=2 passenger detections as overload with estimated riders = passenger boxes + rider.
            if len(passenger_boxes) >= 2:
                vehicle_number = _nearest_plate_number_for_boxes(
                    passenger_boxes, number_plates, img, ocr
                )
                obox_key = ("overload_by_count", vehicle_number)
                if obox_key not in overload_keys:
                    overload_keys.add(obox_key)
                    est_riders = max(3, len(passenger_boxes) + 1)
                    cvzone.putTextRect(
                        img,
                        f"Overloaded: {est_riders} Passengers",
                        (20, 120),
                        scale=1.2,
                        offset=10,
                        thickness=2,
                        colorR=(255, 0, 0),
                    )
                    _append_violation({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'violation_type': 'Overloaded Vehicle',
                        'vehicle_number': vehicle_number,
                        'passenger_count': est_riders,
                        'fine': FINE_INR,
                        '_anchor': _box_center(passenger_boxes[0]),
                    })

            # Fallback: 3+ helmet-class heads in frame but ROI either missing or under-counts (once per video)
            had_roi_overload = any(
                _count_riders_in_passenger_box(pb, helmet_detections) > 2 for pb in passenger_boxes
            )
            if (
                len(helmet_detections) > 2
                and not had_roi_overload
                and ("global_triple",) not in overload_keys
            ):
                overload_keys.add(("global_triple",))
                n_riders = len(helmet_detections)
                vehicle_number = _nearest_plate_number_for_boxes(
                    [(h[0], h[1], h[2], h[3]) for h in helmet_detections],
                    number_plates,
                    img,
                    ocr,
                )
                cvzone.putTextRect(
                    img,
                    f"Overloaded: {n_riders} Passengers (estimate)",
                    (20, 80),
                    scale=1.2,
                    offset=10,
                    thickness=2,
                    colorR=(255, 0, 0),
                )
                _append_violation({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'violation_type': 'Overloaded Vehicle',
                    'vehicle_number': vehicle_number,
                    'passenger_count': n_riders,
                    'fine': FINE_INR,
                })

            # Collect plate OCR reads for later UNREAD backfill.
            if frame_count % PLATE_OCR_SAMPLE_STRIDE == 0:
                for px1, py1, px2, py2 in number_plates:
                    crop = img[py1:py2, px1:px2]
                    if crop.size == 0:
                        continue
                    plate_text, plate_score = predict_number_plate(crop, ocr)
                    if plate_text:
                        plate_history.append(
                            {
                                "center": _box_center((px1, py1, px2, py2)),
                                "text": plate_text,
                                "score": float(plate_score or 0.0),
                            }
                        )
                if len(plate_history) > MAX_PLATE_HISTORY:
                    plate_history = plate_history[-MAX_PLATE_HISTORY:]
            
            # Write frame to output video
            output.write(img)
            
            _report_progress()
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}")
            
            # Optional preview window (skip if OpenCV is headless)
            if not preview_failed:
                try:
                    cv2.imshow('Video Processing', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    preview_failed = True
        
        # Cleanup
        cap.release()
        output.release()
        try:
            if enable_preview:
                cv2.destroyAllWindows()
        except Exception:
            pass
        
        if violations:
            try:
                _resolve_unread_plates(violations, plate_history)
                for v in violations:
                    v.pop("_anchor", None)
                _attach_consolidated_tickets(violations, timestamp)
            except Exception as pdf_err:
                print(f"E-ticket PDF failed: {pdf_err}")
                traceback.print_exc()

        if callable(progress_cb):
            try:
                progress_cb(frame_count=frame_count, total_frames=total_frames, percent=100)
            except Exception:
                pass

        return output_video_path, violations
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    process_video(str(REPO_ROOT / "data" / "videos" / "22.mp4"))

