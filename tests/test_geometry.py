"""Numerical and event-level regressions for calibrated traffic measurements."""

import numpy as np
import pytest

from utils.homography_utils import HomographyTransformer
from utils.speed_estimation import SpeedEstimator
from utils.violation_check import ViolationDetector


SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]
LINE = [[0, 50], [100, 50]]


def box_at(x, y):
    return [x - 2, y - 4, x + 2, y]


def crossing(detector, points, colours=None, track=1, line=LINE):
    if colours is None:
        colours = ["red"] * len(points)
    return [
        detector.check_traffic_light_violation(track, box_at(*point), line, colour)
        for point, colour in zip(points, colours, strict=True)
    ]


def test_measured_rectangle_and_round_trip_preserve_fractional_coordinates():
    transform = HomographyTransformer(SQUARE, 20, 10)
    assert transform.transform_points(SQUARE) == pytest.approx(
        np.array([[0, 0], [20, 0], [20, 10], [0, 10]])
    )
    ground = transform.transform_points([[12.5, 33.3]])
    assert ground == pytest.approx(np.array([[2.5, 3.33]]))
    assert transform.inverse_transform_points(ground) == pytest.approx(np.array([[12.5, 33.3]]))
    assert transform.transform_points([]).shape == (0, 2)


def test_speed_applies_homography_once_and_uses_elapsed_source_frames():
    estimator = SpeedEstimator(SQUARE, 20, 10, fps=30)
    assert estimator.calculate_speed([[0, 0], [50, 0]], 30) == pytest.approx(36)
    # Same two observations over two seconds, despite missing detections.
    assert estimator.calculate_speed([[0, 0], [50, 0]], 60) == pytest.approx(18)
    assert estimator.calculate_speed([[30, 40], [30, 40]], 30) == 0
    assert estimator.calculate_speed([], 30) == 0
    assert estimator.calculate_speed([[30, 40]], 30) == 0


def test_nonrectangular_calibration_maps_ground_motion_before_speed():
    source = [[25, 10], [75, 10], [100, 100], [0, 100]]
    estimator = SpeedEstimator(source, 10, 20, fps=25)
    pixels = estimator.inverse_transform_points([[2, 5], [2, 15]])
    assert estimator.calculate_speed(pixels, 25) == pytest.approx(36)


@pytest.mark.parametrize("elapsed", [0, -1, np.nan, np.inf, True])
def test_invalid_elapsed_time_is_rejected(elapsed):
    estimator = SpeedEstimator(SQUARE, 20, 10, fps=30)
    with pytest.raises(ValueError):
        estimator.calculate_speed([[0, 0], [50, 0]], elapsed)


@pytest.mark.parametrize("fps", [0, -1, np.nan, np.inf, True])
def test_invalid_fps_is_rejected(fps):
    with pytest.raises(ValueError):
        SpeedEstimator(SQUARE, 20, 10, fps=fps)


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [1, 1], [2, 2], [3, 3]],
        [[0, 0], [100, 100], [100, 0], [0, 100]],
        [[0, 0], [100, 0], [100, 0], [0, 100]],
        [[0, 0], [100, 0], [np.nan, 100], [0, 100]],
        [[0, 0], [100, 0], [100, 100]],
    ],
)
def test_invalid_calibration_is_rejected(points):
    with pytest.raises(ValueError):
        HomographyTransformer(points, 20, 10)


def test_projection_rejects_bad_points_and_ground_dimensions():
    transform = HomographyTransformer(SQUARE, 20, 10)
    for points in ([[np.inf, 0]], [[0, 0, 0]], [1, 2]):
        with pytest.raises(ValueError):
            transform.transform_points(points)
    with pytest.raises(ValueError):
        HomographyTransformer(SQUARE, 0, 10)
    with pytest.raises(ValueError):
        SpeedEstimator(SQUARE, 20, 10, 30, meter_per_pixel=0)


def test_birds_eye_rendering_has_separate_pixel_scale():
    transform = HomographyTransformer(SQUARE, 20, 10)
    image = np.zeros((101, 101, 3), dtype=np.uint8)
    image[50, 50] = [20, 100, 200]
    rendered = transform.transform_image(image, pixels_per_unit=5)
    assert rendered.shape == (51, 101, 3)
    assert rendered[25, 50].tolist() == [20, 100, 200]
    restored = transform.inverse_transform_image(rendered, image.shape, pixels_per_unit=5)
    assert restored.shape == image.shape
    assert restored[50, 50].tolist() == [20, 100, 200]
    assert transform.transform_point([50, 50]) == pytest.approx([10, 5])


def test_direct_red_crossing_is_deduplicated_and_approach_is_not_crossing():
    detector = ViolationDetector()
    assert crossing(detector, [(50, 40), (50, 45), (50, 60), (50, 40), (50, 60)]) == [
        False,
        False,
        True,
        False,
        False,
    ]


def test_waiting_on_line_is_not_crossing_until_departure_on_opposite_side():
    detector = ViolationDetector()
    assert crossing(detector, [(50, 40), (50, 50), (50, 50), (50, 60)]) == [
        False,
        False,
        False,
        True,
    ]
    assert crossing(ViolationDetector(), [(50, 40), (50, 50), (50, 40)]) == [False] * 3
    assert crossing(ViolationDetector(), [(50, 50), (50, 60)]) == [False] * 2


def test_crossing_line_extension_and_departing_beyond_endpoint_are_not_events():
    assert crossing(ViolationDetector(), [(150, 40), (150, 60)]) == [False, False]
    assert crossing(ViolationDetector(), [(50, 40), (50, 50), (150, 50), (150, 60)]) == [
        False,
        False,
        False,
        False,
    ]
    assert crossing(ViolationDetector(), [(100, 40), (100, 60)]) == [False, True]


def test_direction_can_be_reversed_and_reverse_traffic_is_ignored():
    assert crossing(ViolationDetector(), [(50, 60), (50, 40)]) == [False, False]
    assert crossing(ViolationDetector(crossing_direction=-1), [(50, 60), (50, 40)]) == [False, True]


@pytest.mark.parametrize("colour", ["green", "yellow", "unknown", None])
def test_nonred_light_does_not_create_later_retroactive_event(colour):
    assert crossing(
        ViolationDetector(), [(50, 40), (50, 60), (50, 65)], ["red", colour, "red"]
    ) == [False, False, False]


def test_changed_line_resets_geometry_and_track_discard_resets_both_event_types():
    detector = ViolationDetector()
    assert crossing(detector, [(50, 40), (50, 60)]) == [False, True]
    assert detector.check_speed_violation(1, 60, "car") is True
    assert detector.check_speed_violation(1, 65, "car") is False
    detector.discard_track(1)
    assert crossing(detector, [(50, 40), (50, 60)]) == [False, True]
    assert detector.check_speed_violation(1, 60, "car") is True
    other = ViolationDetector()
    assert crossing(other, [(50, 40)]) == [False]
    assert crossing(other, [(50, 60)], line=[[0, 55], [100, 55]]) == [False]


def test_threshold_is_strict_and_constructor_does_not_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    detector = ViolationDetector(speed_limits={"car": 30, "default": 20})
    assert detector.check_speed_violation(1, 30, "car") is False
    assert detector.check_speed_violation(1, 30.1, "car") is True
    assert detector.check_speed_violation(1, 40, "car") is False
    assert detector.check_speed_violation(2, 21, "unlisted-class") is True
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("speed", [np.nan, np.inf, -1, True, "invalid"])
def test_invalid_speed_is_rejected(speed):
    with pytest.raises(ValueError):
        ViolationDetector().check_speed_violation(1, speed, "car")


def test_invalid_stop_line_box_and_configuration_are_rejected():
    detector = ViolationDetector()
    with pytest.raises(ValueError):
        detector.check_traffic_light_violation(1, box_at(50, 40), [[0, 0], [0, 0]], "red")
    with pytest.raises(ValueError):
        detector.check_traffic_light_violation(1, [2, 2, 1, 1], LINE, "red")
    with pytest.raises(ValueError):
        ViolationDetector(crossing_direction=0)
    with pytest.raises(ValueError):
        ViolationDetector(speed_limits={"car": -1})
