"""Offline regression checks using synthetic pixels and fictitious OCR output."""

import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from utils.color_detection import TrafficLightColorDetector
from utils.database import TrafficDatabase
from utils.ocr import clean_plate_text, preprocess_plate_image, read_license_plate


def crop(color=(255, 255, 255)):
    return np.full((24, 72, 3), color, dtype=np.uint8)


def fake_ocr(monkeypatch, texts, confidences, lines=None):
    data = {"text": texts, "conf": confidences}
    if lines is not None:
        data.update(
            page_num=[1] * len(texts),
            block_num=[1] * len(texts),
            par_num=[1] * len(texts),
            line_num=lines,
        )
    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        SimpleNamespace(
            Output=SimpleNamespace(DICT="dict"),
            image_to_data=lambda *args, **kwargs: data,
        ),
    )


def test_ocr_joins_line_tokens_with_mixed_confidence_types(monkeypatch):
    # Fabricated OCR tokens, unrelated to a captured vehicle or person.
    fake_ocr(monkeypatch, ["", "01", "ABC", "123"], [-1, 91, "89.5", "92"], [0, 1, 1, 1])
    result = read_license_plate(crop())
    assert result == {"plate_text": "01ABC123", "city": "Adana", "confidence": 89.5}


def test_ocr_keeps_lines_separate_and_rejects_extra_tokens(monkeypatch):
    fake_ocr(monkeypatch, ["01", "ABC123"], [90, 91], [1, 2])
    assert read_license_plate(crop())["plate_text"] is None
    fake_ocr(monkeypatch, ["01ABC123", "EXTRA"], [90, 90], [1, 1])
    assert read_license_plate(crop())["plate_text"] is None


@pytest.mark.parametrize("bad", [-1, "bad", "nan", "inf", 101, None])
def test_ocr_never_drops_a_bad_token_to_manufacture_a_plate(monkeypatch, bad):
    fake_ocr(monkeypatch, ["01ABC123", "EXTRA"], [90, bad], [1, 1])
    assert read_license_plate(crop()) == {"plate_text": None, "city": None, "confidence": 0.0}


def test_ocr_does_not_invent_confidence_for_bad_plate(monkeypatch):
    fake_ocr(monkeypatch, ["NOTAPLATE"], [99], [1])
    assert read_license_plate(crop())["confidence"] == 0


@pytest.mark.parametrize(
    "value",
    [
        "0ABC123",
        "00ABC123",
        "82ABC123",
        "XX01ABC123",
        "01ABC123extra",
        "01ABC123!",
        "01ABCD123",
        None,
    ],
)
def test_plate_filter_requires_complete_candidate(value):
    assert clean_plate_text(value) is None


def test_plate_whitespace_normalization_and_crop_validation():
    assert clean_plate_text("01 abc 123") == "01ABC123"
    for invalid in [None, np.empty((0, 3), dtype=np.uint8), np.ones((2, 2), dtype=float)]:
        with pytest.raises(ValueError):
            preprocess_plate_image(invalid)
    assert preprocess_plate_image(crop()).shape == (24, 72)


@pytest.mark.parametrize(
    "bgr,expected",
    [
        ((0, 0, 255), "red"),
        ((0, 255, 255), "yellow"),
        ((0, 255, 0), "green"),
        ((0, 0, 0), "unknown"),
    ],
)
def test_known_light_colors(bgr, expected):
    assert TrafficLightColorDetector().detect_color(crop(bgr)) == expected


def test_ambiguous_or_invalid_light_is_unknown():
    detector = TrafficLightColorDetector()
    mixed = crop((0, 0, 255))
    mixed[:, :36] = (0, 255, 0)
    assert detector.detect_color(mixed) == "unknown"
    for invalid in [None, np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((2, 2))]:
        assert detector.detect_color(invalid) == "unknown"
    for box in [[0, 0, 0, 2], [-10, 0, -2, 3], [0, 0, float("nan"), 4], [0, 2]]:
        assert detector.analyze_traffic_light(crop(), box)[0] == "unknown"


def test_light_crop_is_clipped_and_input_not_modified():
    image = crop((0, 0, 255))
    before = image.copy()
    color, annotated = TrafficLightColorDetector().analyze_traffic_light(
        image, [-1.5, -2, 20.5, 15]
    )
    assert color == "red"
    assert np.array_equal(image, before)
    assert annotated is not image


def test_database_discards_personal_fields_by_default_and_upserts(tmp_path):
    with TrafficDatabase(tmp_path / "run" / "events.db") as database:
        identifier = database.add_vehicle(7, "car", "SYNTHETIC", "DEMO", 99)
        assert database.add_vehicle(7, "car", "OTHER", "DEMO", 98) == identifier
        assert database.conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0] == 1
        vehicle = database.get_vehicle_by_tracker_id(7)
        assert vehicle["plate_text"] is None and vehicle["city"] is None
        assert database.conn.execute("SELECT confidence FROM vehicles").fetchone()[0] is None
        event = database.add_violation(
            identifier,
            1,
            speed=18,
            speed_limit=15,
            evidence_image="PRIVATE",
            evidence_text="PRIVATE",
        )
        rows = database.get_all_violations()
        assert rows[0]["id"] == event and rows[0]["speed"] == 18
        assert rows[0]["evidence_image"] is None and rows[0]["evidence_text"] is None
        assert database.get_vehicle_by_id(identifier) == vehicle
        assert len(database.get_violations_by_vehicle_id(identifier)) == 1
        with pytest.raises(sqlite3.IntegrityError):
            database.add_violation(999, 1)
        with pytest.raises(sqlite3.IntegrityError):
            database.conn.execute("INSERT INTO vehicles(tracker_id) VALUES (7)")
        with pytest.raises(ValueError):
            database.get_all_violations(-1)
    assert database.conn is None
    database.close()


def test_database_supports_bare_filename_and_explicit_opt_in(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with TrafficDatabase("events.db", allow_personal_data=True) as database:
        vehicle = database.add_vehicle(1, "car", "SYNTHETIC", "DEMO", 80)
        database.add_violation(vehicle, 1, evidence_text="SYNTHETIC ONLY")
        assert database.get_all_violations()[0]["plate_text"] == "SYNTHETIC"
        assert database.get_all_violations()[0]["evidence_text"] == "SYNTHETIC ONLY"
    with TrafficDatabase(":memory:") as database:
        assert database.get_all_violations() == []
