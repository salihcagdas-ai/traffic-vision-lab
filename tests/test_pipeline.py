import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np
import pytest

import main
from main import Analyzer, Calibration, Detection, demo_calibration, run_demo, run_video
from utils.database import TrafficDatabase


def test_demo_video_database_and_known_motion(tmp_path):
    output = tmp_path / "demo"
    summary = run_demo(output)
    assert summary["mode"] == "synthetic-demo"
    assert summary["frames"] == 90
    assert summary["tracks_in_region"] == 2
    # 300 pixel displacement / 20 px per metre over 89 / 30 seconds.
    assert summary["peak_estimated_speed_kmh"] == pytest.approx(15 / (89 / 30) * 3.6, abs=0.001)
    assert [(e["track_id"], e["kind"]) for e in summary["events"]] == [
        (1, "speed_threshold"),
        (1, "red_light"),
    ]
    assert not summary["personal_data_saved"]
    assert json.loads((output / "summary.json").read_text()) == summary
    cap = cv2.VideoCapture(str(output / "demo.mp4"))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    assert count == 90
    with sqlite3.connect(output / "events.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM violations").fetchone()[0] == 2
        assert db.execute("SELECT violation_type FROM violations ORDER BY id").fetchall() == [
            (3,),
            (1,),
        ]
        assert (
            db.execute(
                "SELECT COUNT(*) FROM vehicles WHERE plate_text IS NOT NULL OR city IS NOT NULL"
            ).fetchone()[0]
            == 0
        )
    assert (output / "preview.png").read_bytes().startswith(b"\x89PNG")
    with pytest.raises(FileExistsError):
        run_demo(output)


def test_demo_is_deterministic_and_configuration_matches():
    first, d1 = main.demo_frame(50)
    second, d2 = main.demo_frame(50)
    assert np.array_equal(first, second) and d1 == d2
    config = Calibration.from_dict(json.loads(Path("examples/calibration.demo.json").read_text()))
    assert np.array_equal(config.source_points, demo_calibration().source_points)


def test_actual_frame_gaps_and_roi_exit_reset():
    config = demo_calibration()
    with TrafficDatabase(":memory:") as db:
        analyzer = Analyzer(config, 30, db)
        analyzer.process(0, [Detection(1, (300, 150, 350, 180))], "green")
        speeds = analyzer.process(15, [Detection(1, (300, 200, 350, 230))], "green")
        assert speeds[1] == pytest.approx(2.5 / 0.5 * 3.6)
        analyzer.process(16, [Detection(1, (0, 0, 20, 20))], "red")
        assert 1 not in analyzer.history
        assert analyzer.process(17, [Detection(1, (300, 300, 350, 350))], "red") == {}
        assert not any(event["kind"] == "red_light" for event in analyzer.events)


def test_history_expires_and_duplicate_frames_fail():
    with TrafficDatabase(":memory:") as db:
        analyzer = Analyzer(demo_calibration(), 30, db)
        analyzer.process(0, [Detection(1, (300, 150, 350, 180))], "red")
        analyzer.process(60, [], "unknown")
        assert not analyzer.history
        with pytest.raises(ValueError, match="strictly increasing"):
            analyzer.process(60, [], "red")


@pytest.mark.parametrize(
    "field,value",
    [
        ("width_m", 0),
        ("height_m", float("nan")),
        ("crossing_direction", 0),
        ("light_roi", [-1, 0, 4, 5]),
        ("image_size", [960.5, 540]),
        ("stop_line", [[2, 2], [2, 2]]),
        ("speed_limits_kmh", {"car": 0}),
        ("source_points", [[0, 0]] * 4),
    ],
)
def test_bad_calibration(field, value):
    data = json.loads(Path("examples/calibration.demo.json").read_text())
    data[field] = value
    with pytest.raises(ValueError):
        Calibration.from_dict(data)


def test_missing_model_is_error_never_demo(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "run_demo", lambda *args: pytest.fail("silent demo fallback"))
    with pytest.raises(ValueError, match="local file"):
        main.YoloBackend(tmp_path / "missing.pt", 0.3, "cpu")
    with pytest.raises(ValueError):
        main.local_file("https://example.invalid/video.mp4")


def test_video_path_with_injected_detector_saves_metadata_only(tmp_path, monkeypatch):
    video = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (960, 540))
    assert writer.isOpened()
    for index in range(3):
        writer.write(main.demo_frame(index)[0])
    writer.release()
    weights = tmp_path / "fixture.pt"
    weights.write_bytes(b"unused: injected backend")

    class Backend:
        def __init__(self, *args):
            pass

        def tracks(self, frame, classes):
            return [Detection(1, (300, 150, 350, 180))]

    monkeypatch.setattr(main, "YoloBackend", Backend)
    output = tmp_path / "analysis"
    report = run_video(video, weights, Path("examples/calibration.demo.json"), output)
    assert report["mode"] == "local-video" and report["frames"] == 3
    assert sorted(p.name for p in output.iterdir()) == ["events.sqlite3", "summary.json"]
    with pytest.raises(ValueError, match="already exists"):
        run_video(video, weights, Path("examples/calibration.demo.json"), output)


def test_video_size_mismatch_fails_before_backend_loading(tmp_path, monkeypatch):
    video = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 64))
    writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()
    weights = tmp_path / "fixture.pt"
    weights.touch()
    monkeypatch.setattr(main, "YoloBackend", lambda *a: pytest.fail("backend should not load"))
    with pytest.raises(ValueError, match="image_size"):
        run_video(video, weights, Path("examples/calibration.demo.json"), tmp_path / "output")
    assert not (tmp_path / "output").exists()
