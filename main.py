"""Traffic Vision Lab: explicit synthetic demo or local video analysis.

Portfolio refactor of a shared graduation project; see NOTICE.md.
No camera/URL input, implicit model download, or real-frame export.
"""

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

from utils.color_detection import TrafficLightColorDetector
from utils.database import TrafficDatabase
from utils.speed_estimation import SpeedEstimator
from utils.violation_check import ViolationDetector


@dataclass(frozen=True)
class Detection:
    track_id: int
    box: tuple
    vehicle_class: str = "car"
    confidence: float = 1.0


@dataclass(frozen=True)
class Calibration:
    image_size: tuple
    source_points: np.ndarray
    width_m: float
    height_m: float
    stop_line: np.ndarray
    light_roi: tuple
    speed_limits: dict
    crossing_direction: int = 1

    @classmethod
    def from_dict(cls, data):
        size = data["image_size"]
        if len(size) != 2 or any(type(n) is not int or n <= 0 for n in size):
            raise ValueError("image_size must contain positive integer width and height")
        points = np.asarray(data["source_points"], dtype=np.float32)
        line = np.asarray(data["stop_line"], dtype=np.float32)
        roi = data["light_roi"]
        if points.shape != (4, 2) or line.shape != (2, 2):
            raise ValueError("source_points needs four corners; stop_line needs two endpoints")
        if not np.isfinite(points).all() or not np.isfinite(line).all():
            raise ValueError("calibration coordinates must be finite")
        for coords in (points, line):
            if (
                (coords < 0).any()
                or (coords[:, 0] >= size[0]).any()
                or (coords[:, 1] >= size[1]).any()
            ):
                raise ValueError("calibration coordinates must be inside the image")
        if np.linalg.norm(line[1] - line[0]) <= 1e-6:
            raise ValueError("stop_line endpoints must be distinct")
        if len(roi) != 4 or any(type(n) is not int for n in roi):
            raise ValueError("light_roi must contain four integer pixel coordinates")
        if not (0 <= roi[0] < roi[2] <= size[0] and 0 <= roi[1] < roi[3] <= size[1]):
            raise ValueError("light_roi must be a nonempty rectangle inside the image")
        limits = data["speed_limits_kmh"]
        if (
            not isinstance(limits, dict)
            or not limits
            or any(
                not isinstance(k, str)
                or not k
                or isinstance(v, bool)
                or not math.isfinite(float(v))
                or float(v) <= 0
                for k, v in limits.items()
            )
        ):
            raise ValueError("speed_limits_kmh must map class names to positive thresholds")
        direction = data.get("crossing_direction", 1)
        if type(direction) is not int or direction not in (-1, 1):
            raise ValueError("crossing_direction must be -1 or 1")
        # Constructor validates the quadrilateral and positive physical dimensions.
        SpeedEstimator(points, data["width_m"], data["height_m"], 1)
        return cls(
            tuple(size),
            points,
            float(data["width_m"]),
            float(data["height_m"]),
            line,
            tuple(roi),
            {k: float(v) for k, v in limits.items()},
            direction,
        )


class Analyzer:
    """One camera plane and signal, scoped to the calibrated quadrilateral."""

    def __init__(self, calibration, fps, database):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be positive and finite")
        self.calibration = calibration
        self.fps = fps
        self.database = database
        self.speed = SpeedEstimator(
            calibration.source_points, calibration.width_m, calibration.height_m, fps
        )
        self.violations = ViolationDetector(
            speed_limits=calibration.speed_limits, crossing_direction=calibration.crossing_direction
        )
        self.history = {}
        self.last_seen = {}
        self.last_frame = -1
        self.events = []
        self.vehicle_ids = set()
        self.peak_speed = 0.0

    def process(self, frame_index, detections, light_color):
        if type(frame_index) is not int or frame_index <= self.last_frame:
            raise ValueError("frame indices must be strictly increasing nonnegative integers")
        self.last_frame = frame_index
        # Expire stale tracks; a new track must build its own motion history.
        for track_id in list(self.last_seen):
            if frame_index - self.last_seen[track_id] > self.fps:
                self._discard(track_id)
        speeds = {}
        seen = set()
        for detection in detections:
            if detection.vehicle_class not in self.calibration.speed_limits:
                continue
            if detection.track_id in seen:
                raise ValueError("duplicate track ID in one frame")
            seen.add(detection.track_id)
            box = np.asarray(detection.box, dtype=float)
            if (
                box.shape != (4,)
                or not np.isfinite(box).all()
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                raise ValueError("detections must contain finite ordered xyxy boxes")
            if (
                detection.track_id < 0
                or not math.isfinite(detection.confidence)
                or not 0 <= detection.confidence <= 1
            ):
                raise ValueError("invalid track ID or detection confidence")
            point = ((box[0] + box[2]) / 2, box[3])
            if cv2.pointPolygonTest(self.calibration.source_points, point, False) < 0:
                self._discard(detection.track_id)
                continue
            history = self.history.setdefault(detection.track_id, deque())
            history.append((frame_index, point))
            while len(history) > 2 and frame_index - history[0][0] > self.fps:
                history.popleft()
            self.last_seen[detection.track_id] = frame_index
            self.vehicle_ids.add(detection.track_id)
            vehicle_id = self.database.add_vehicle(
                detection.track_id, vehicle_class=detection.vehicle_class
            )
            speed = None
            elapsed = history[-1][0] - history[0][0]
            if elapsed >= max(1, math.ceil(0.25 * self.fps)):
                # Pixel coordinates enter this method once. Actual frame gap supplies time.
                speed = self.speed.calculate_speed([p for _, p in history], elapsed)
                speeds[detection.track_id] = speed
                self.peak_speed = max(self.peak_speed, speed)
            if self.violations.check_traffic_light_violation(
                detection.track_id, box, self.calibration.stop_line, light_color
            ):
                self._event(vehicle_id, detection, frame_index, "red_light", 1, speed, light_color)
            if speed is not None and self.violations.check_speed_violation(
                detection.track_id, speed, detection.vehicle_class
            ):
                self._event(
                    vehicle_id,
                    detection,
                    frame_index,
                    "speed_threshold",
                    self.violations.VIOLATION_TYPES["SPEEDING"],
                    speed,
                    light_color,
                )
        return speeds

    def _discard(self, track_id):
        self.history.pop(track_id, None)
        self.last_seen.pop(track_id, None)
        self.violations.discard_track(track_id)

    def _event(self, vehicle_id, detection, frame, kind, code, speed, light):
        limit = self.calibration.speed_limits.get(detection.vehicle_class)
        self.database.add_violation(
            vehicle_id, code, speed=speed, speed_limit=limit, light_color=light
        )
        self.events.append(
            {
                "track_id": detection.track_id,
                "kind": kind,
                "time_seconds": round(frame / self.fps, 3),
                "speed_kmh": None if speed is None else round(speed, 3),
            }
        )

    def summary(self, mode, frames):
        return {
            "mode": mode,
            "frames": frames,
            "fps": self.fps,
            "tracks_in_region": len(self.vehicle_ids),
            "events": self.events,
            "peak_estimated_speed_kmh": round(self.peak_speed, 3),
            "personal_data_saved": False,
            "real_frames_saved": False,
            "accuracy_benchmark": False,
        }


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_summary(output, summary):
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def demo_calibration():
    return Calibration.from_dict(
        {
            "image_size": [960, 540],
            "source_points": [[220, 140], [740, 140], [740, 500], [220, 500]],
            "width_m": 26,
            "height_m": 18,
            "stop_line": [[220, 335], [740, 335]],
            "light_roi": [810, 140, 855, 185],
            "speed_limits_kmh": {"car": 12},
            "crossing_direction": 1,
        }
    )


def demo_frame(index, total=90):
    """Known geometric trajectories; these boxes are NOT neural model detections."""
    frame = np.full((540, 960, 3), (25, 22, 17), dtype=np.uint8)
    cv2.rectangle(frame, (220, 140), (740, 500), (48, 43, 35), -1)
    cv2.rectangle(frame, (220, 140), (740, 500), (125, 119, 103), 2)
    for y in range(150, 500, 55):
        cv2.line(frame, (480, y), (480, min(y + 25, 500)), (130, 125, 112), 2)
    cv2.line(frame, (220, 335), (740, 335), (85, 165, 255), 3)
    cv2.rectangle(frame, (810, 140), (854, 184), (0, 0, 255), -1)
    cv2.putText(frame, "RED", (810, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 170, 255), 1)
    cv2.putText(
        frame, "TRAFFIC VISION LAB", (36, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 240, 245), 2
    )
    cv2.putText(
        frame,
        "SYNTHETIC DEMO | NO MODEL INFERENCE",
        (36, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (150, 205, 230),
        1,
    )
    cv2.putText(
        frame,
        "Known trajectories / measured time / calibrated plane",
        (36, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (175, 175, 175),
        1,
    )
    progress = index / (total - 1)
    detections = []
    for track_id, x, bottom, color in [
        (1, 350, 170 + 300 * progress, (225, 190, 90)),
        (2, 610, 470 - 120 * progress, (160, 180, 140)),
    ]:
        box = (x - 25, bottom - 44, x + 25, bottom)
        detections.append(Detection(track_id, box))
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, -1)
        cv2.putText(
            frame,
            f"track {track_id}",
            (x - 32, int(bottom) - 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (225, 225, 225),
            1,
        )
    cv2.putText(
        frame,
        "Synthetic shapes only. No people, plates or road footage.",
        (36, 525),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (175, 175, 175),
        1,
    )
    return frame, detections


def run_demo(output):
    output = new_output(output)
    calibration, fps, total = demo_calibration(), 30.0, 90
    database = TrafficDatabase(str(output / "events.sqlite3"))
    writer = None
    try:
        writer = cv2.VideoWriter(
            str(output / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540)
        )
        if not writer.isOpened():
            raise RuntimeError("MP4 writer is unavailable")
        analyzer = Analyzer(calibration, fps, database)
        colors = TrafficLightColorDetector()
        for index in range(total):
            frame, detections = demo_frame(index, total)
            x1, y1, x2, y2 = calibration.light_roi
            light = colors.detect_color(frame[y1:y2, x1:x2])
            speeds = analyzer.process(index, detections, light)
            for row, (track_id, speed) in enumerate(sorted(speeds.items())):
                cv2.putText(
                    frame,
                    f"Track {track_id}: {speed:.1f} km/h",
                    (32, 170 + row * 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (220, 220, 220),
                    1,
                )
            cv2.putText(
                frame,
                f"Events: {len(analyzer.events)}",
                (770, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 220, 220),
                1,
            )
            writer.write(frame)
            if index == 60:
                ok, png = cv2.imencode(".png", frame)
                if not ok:
                    raise RuntimeError("PNG encoding failed")
                (output / "preview.png").write_bytes(png.tobytes())
    finally:
        if writer is not None:
            writer.release()
        database.close()
    return write_summary(output, analyzer.summary("synthetic-demo", total))


def local_file(value, suffixes=None):
    path = Path(value)
    if "://" in str(value) or not path.is_file():
        raise ValueError(
            "A readable local file is required; URLs and camera indices are unsupported"
        )
    if suffixes and path.suffix.lower() not in suffixes:
        raise ValueError("Unsupported local file extension")
    return path.resolve()


class YoloBackend:
    def __init__(self, weights, confidence, device):
        weights = local_file(weights, {".pt"})
        # Ultralytics performs no package auto-install/update or telemetry in this runner.
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_VERBOSE"] = "false"
        self._settings_dir = tempfile.TemporaryDirectory(prefix="traffic-vision-settings-")
        os.environ["YOLO_CONFIG_DIR"] = self._settings_dir.name
        try:
            from ultralytics import YOLO, settings
        except ImportError as exc:
            raise RuntimeError("Install requirements-models.txt for video mode") from exc
        settings.update({"sync": False})
        self.model = YOLO(str(weights), task="detect")
        self.plate_model = None
        self.plate_reads = 0
        self.confidence, self.device = confidence, device

    def enable_plate_reader(self, weights):
        weights = local_file(weights, {".pt"})
        if shutil.which("tesseract") is None:
            raise RuntimeError("Optional plate reading requires Tesseract on PATH")
        try:
            import pytesseract  # noqa: F401
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install requirements-models.txt for plate reading") from exc
        self.plate_model = YOLO(str(weights), task="detect")

    def read_plates(self, frame):
        """Count successful OCR observations; discard text and never save crops."""
        if self.plate_model is None:
            return
        from utils.ocr import read_license_plate

        result = self.plate_model.predict(
            frame, conf=self.confidence, device=self.device, verbose=False, save=False
        )[0]
        if result.boxes is None:
            return
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box.astype(int)
            height, width = frame.shape[:2]
            x1, x2 = np.clip([x1, x2], 0, width)
            y1, y2 = np.clip([y1, y2], 0, height)
            crop = frame[y1:y2, x1:x2]
            if crop.size and read_license_plate(crop)["plate_text"]:
                self.plate_reads += 1

    def tracks(self, frame, allowed_classes):
        class_ids = [key for key, value in self.model.names.items() if value in allowed_classes]
        if not class_ids:
            raise ValueError("Model class names do not match the calibrated speed-limit classes")
        result = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=class_ids,
            conf=self.confidence,
            device=self.device,
            verbose=False,
            save=False,
            show=False,
        )[0]
        if result.boxes is None or result.boxes.id is None:
            return []
        boxes = result.boxes
        return [
            Detection(int(track_id), tuple(box), self.model.names[int(class_id)], float(conf))
            for box, track_id, class_id, conf in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.id.cpu().numpy(),
                boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy(),
                strict=True,
            )
        ]


def run_video(
    input_path, weights, calibration_path, output, confidence=0.35, device="cpu", plate_weights=None
):
    input_path = local_file(input_path, {".mp4", ".avi", ".mov", ".mkv"})
    weights = local_file(weights, {".pt"})
    if plate_weights is not None:
        plate_weights = local_file(plate_weights, {".pt"})
    config_path = local_file(calibration_path, {".json"})
    calibration = Calibration.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    if not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError("confidence must be in (0,1]")
    if Path(output).exists():
        raise ValueError("Output directory already exists; choose a new run directory")
    capture = cv2.VideoCapture(str(input_path))
    database = None
    try:
        if not capture.isOpened():
            raise ValueError("Video cannot be opened")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video must have valid constant-frame-rate metadata")
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Video contains no readable frames")
        if (frame.shape[1], frame.shape[0]) != calibration.image_size:
            raise ValueError("Calibration image_size must exactly match the input video")
        backend = YoloBackend(weights, confidence, device)
        if plate_weights is not None:
            backend.enable_plate_reader(plate_weights)
        output = new_output(output)
        database = TrafficDatabase(str(output / "events.sqlite3"))
        analyzer = Analyzer(calibration, fps, database)
        colors = TrafficLightColorDetector()
        index = 0
        while ok:
            if (frame.shape[1], frame.shape[0]) != calibration.image_size:
                raise ValueError("Video frame dimensions changed")
            x1, y1, x2, y2 = calibration.light_roi
            light = colors.detect_color(frame[y1:y2, x1:x2])
            analyzer.process(index, backend.tracks(frame, calibration.speed_limits), light)
            if plate_weights is not None:
                backend.read_plates(frame)
            index += 1
            ok, frame = capture.read()
        summary = analyzer.summary("local-video", index)
        summary["plate_ocr_enabled"] = plate_weights is not None
        summary["plate_ocr_observations"] = backend.plate_reads if plate_weights is not None else 0
        return write_summary(output, summary)
    finally:
        capture.release()
        if database is not None:
            database.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Deterministic geometry demo; no trained models or personal data"
    )
    demo.add_argument("--output", type=Path, required=True)
    video = commands.add_parser(
        "video", help="Local vehicle weights + calibrated video; metadata only"
    )
    video.add_argument("--input", type=Path, required=True)
    video.add_argument("--weights", type=Path, required=True)
    video.add_argument("--calibration", type=Path, required=True)
    video.add_argument("--output", type=Path, required=True)
    video.add_argument("--confidence", type=float, default=0.35)
    video.add_argument("--device", default="cpu")
    video.add_argument(
        "--plate-weights", type=Path, help="Optional local plate detector; OCR text is discarded"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            summary = run_demo(args.output)
        else:
            summary = run_video(
                args.input,
                args.weights,
                args.calibration,
                args.output,
                args.confidence,
                args.device,
                args.plate_weights,
            )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, cv2.error) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
