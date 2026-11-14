"""Illustrative traffic events; no image, plate or filesystem persistence."""

import numpy as np

from .homography_utils import finite_points, positive_number


class ViolationDetector:
    """Emit at most one event of each type per active track.

    Speed thresholds are configurable educational values, not legal decisions.
    Red-light events require crossing the finite stop-line segment. Signed side
    is cross(line_end-line_start, vehicle_bottom_center-line_start).
    Direction +1 means negative-to-positive; -1 means positive-to-negative.
    Landing on the line or waiting there is not a confirmed crossing. Leaving
    on the opposite side completes it, using the current light colour.
    Green, yellow and unknown emit nothing but update geometry; later red
    cannot retroactively flag their crossing. discard_track expires state.
    """

    VIOLATION_TYPES = {"RED_LIGHT": 1, "YELLOW_LIGHT": 2, "SPEEDING": 3}

    def __init__(self, speed_limits=None, crossing_direction=1):
        if isinstance(crossing_direction, (bool, np.bool_)) or crossing_direction not in (-1, 1):
            raise ValueError("crossing_direction must be +1 or -1")
        limits = {"car": 50, "truck": 40, "motorcycle": 50, "bus": 50, "default": 50}
        if speed_limits is not None:
            if not isinstance(speed_limits, dict):
                raise ValueError("speed_limits must be a dictionary")
            limits.update(speed_limits)
        self.speed_limits = {
            str(name).lower(): positive_number(value, f"speed limit for {name}")
            for name, value in limits.items()
        }
        self.crossing_direction = int(crossing_direction)
        self._crossings = {}
        self.detected_violations = {}

    @staticmethod
    def _side(point, start, vector, length):
        offset = point - start
        distance = (vector[0] * offset[1] - vector[1] * offset[0]) / length
        return 0 if abs(distance) <= 1e-9 else (1 if distance > 0 else -1)

    @staticmethod
    def _segment_intersects(previous, current, start, vector):
        """Whether motion reaches the finite line, including its endpoints."""
        motion = current - previous
        denominator = motion[0] * vector[1] - motion[1] * vector[0]
        if abs(denominator) <= 1e-12:
            return False
        offset = start - previous
        motion_fraction = (offset[0] * vector[1] - offset[1] * vector[0]) / denominator
        line_fraction = (offset[0] * motion[1] - offset[1] * motion[0]) / denominator
        tolerance = 1e-9
        return (
            -tolerance <= motion_fraction <= 1 + tolerance
            and -tolerance <= line_fraction <= 1 + tolerance
        )

    def _emit(self, vehicle_id, kind):
        emitted = self.detected_violations.setdefault(vehicle_id, set())
        if kind in emitted:
            return False
        emitted.add(kind)
        return True

    def check_traffic_light_violation(
        self, vehicle_id, vehicle_box, cross_line, light_color, plate_text=None
    ):
        """Update geometry and emit a new red crossing; plate input is ignored."""
        box = np.asarray(vehicle_box, dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("vehicle_box must be a finite positive-area XYXY box")
        line = finite_points(cross_line, name="cross_line")
        if line.shape != (2, 2):
            raise ValueError("cross_line must contain two distinct endpoints")
        vector = line[1] - line[0]
        length = float(np.linalg.norm(vector))
        if length <= 1e-12:
            raise ValueError("cross_line endpoints must be distinct")
        point = np.array([(box[0] + box[2]) / 2, box[3]])
        side = self._side(point, line[0], vector, length)
        key = tuple(line.flatten())
        prior = self._crossings.get(vehicle_id)
        # Changed calibration must not join observations from different lines.
        if prior is None or prior[2] != key:
            self._crossings[vehicle_id] = (point, side, key)
            return False
        previous, previous_nonzero_side, _ = prior
        crossed = (
            previous_nonzero_side == -self.crossing_direction
            and side == self.crossing_direction
            and self._segment_intersects(previous, point, line[0], vector)
        )
        self._crossings[vehicle_id] = (point, side or previous_nonzero_side, key)
        return bool(
            crossed
            and str(light_color).lower() == "red"
            and self._emit(vehicle_id, self.VIOLATION_TYPES["RED_LIGHT"])
        )

    def get_speed_limit(self, vehicle_class):
        return self.speed_limits.get(str(vehicle_class).lower(), self.speed_limits["default"])

    def check_speed_violation(self, vehicle_id, speed, vehicle_class, plate_text=None):
        """Emit only the first strict threshold exceedance for this track."""
        try:
            value = float(speed)
        except (TypeError, ValueError) as exc:
            raise ValueError("speed must be a finite non-negative number") from exc
        if isinstance(speed, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
            raise ValueError("speed must be a finite non-negative number")
        return bool(
            value > self.get_speed_limit(vehicle_class)
            and self._emit(vehicle_id, self.VIOLATION_TYPES["SPEEDING"])
        )

    def discard_track(self, vehicle_id):
        """Forget position and emitted events for an expired track."""
        self._crossings.pop(vehicle_id, None)
        self.detected_violations.pop(vehicle_id, None)
