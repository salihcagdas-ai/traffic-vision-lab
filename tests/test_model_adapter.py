"""Optional backend smoke test: local random weights, not an accuracy benchmark."""

import importlib.util
import socket

import numpy as np
import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None, reason="optional model packages"
)
def test_local_model_and_bytetrack_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    monkeypatch.setenv("YOLO_AUTOINSTALL", "false")
    monkeypatch.setenv("YOLO_VERBOSE", "false")
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "settings"))

    def no_network(*args, **kwargs):
        raise AssertionError("The local model smoke test must not access the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    from ultralytics import YOLO

    import main

    # A shipped architecture initializes random parameters; no pretrained weight is fetched.
    model = YOLO("yolo11n.yaml", task="detect")
    model.model.names = {0: "car", **{i: f"unused_{i}" for i in range(1, 80)}}
    weights = tmp_path / "untrained-fixture.pt"
    model.save(str(weights))
    backend = main.YoloBackend(weights, 0.99, "cpu")
    frame = np.zeros((64, 64, 3), np.uint8)
    assert isinstance(backend.tracks(frame, {"car": 12}), list)
    assert isinstance(backend.tracks(frame, {"car": 12}), list)
    with pytest.raises(ValueError, match="class names"):
        backend.tracks(frame, {"class-not-in-model": 10})
    assert weights.is_file()


def test_optional_plate_crops_are_clipped_and_text_is_discarded(monkeypatch):
    from types import SimpleNamespace
    import main
    import utils.ocr

    class Array:
        def cpu(self):
            return self

        def numpy(self):
            return np.array([[-20, -20, -5, -5], [0, 0, 20, 20]], dtype=float)

    backend = main.YoloBackend.__new__(main.YoloBackend)
    backend.plate_model = SimpleNamespace(
        predict=lambda *a, **k: [SimpleNamespace(boxes=SimpleNamespace(xyxy=Array()))]
    )
    backend.confidence, backend.device, backend.plate_reads = 0.3, "cpu", 0
    monkeypatch.setattr(utils.ocr, "read_license_plate", lambda crop: {"plate_text": "34ABC123"})
    backend.read_plates(np.zeros((32, 32, 3), np.uint8))
    assert backend.plate_reads == 1
    assert "plate_text" not in vars(backend)
