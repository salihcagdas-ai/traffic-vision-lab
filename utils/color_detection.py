"""HSV traffic-light heuristic; uncertain or mixed crops return unknown."""

import cv2
import numpy as np


class TrafficLightColorDetector:
    """Classify a configured light crop, not arbitrary road pixels.

    At least 5% of the crop must match a color. A winner must account for
    75% of matched pixels; simultaneous substantial red and green are
    ambiguous. These are heuristic thresholds, not measured model accuracy.
    """

    @staticmethod
    def _valid_image(image):
        return (
            isinstance(image, np.ndarray)
            and image.size > 0
            and image.ndim == 3
            and image.shape[2] == 3
            and image.dtype == np.uint8
        )

    def detect_color(self, image):
        if not self._valid_image(image):
            return "unknown"
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = {
            "red": cv2.bitwise_or(
                cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)),
                cv2.inRange(hsv, (160, 100, 100), (180, 255, 255)),
            ),
            "yellow": cv2.inRange(hsv, (20, 100, 100), (40, 255, 255)),
            "green": cv2.inRange(hsv, (45, 100, 100), (90, 255, 255)),
        }
        counts = {color: cv2.countNonZero(mask) for color, mask in masks.items()}
        area = image.shape[0] * image.shape[1]
        total = sum(counts.values())
        if total < 0.05 * area:
            return "unknown"
        if min(counts["red"], counts["green"]) >= 0.05 * area:
            return "unknown"
        winner = max(counts, key=counts.get)
        return winner if counts[winner] >= 0.75 * total else "unknown"

    def analyze_traffic_light(self, image, traffic_light_box):
        """Return color and an annotated copy; invalid crops return unknown."""
        if not self._valid_image(image):
            return "unknown", image
        result = image.copy()
        try:
            box = np.asarray(traffic_light_box, dtype=float)
        except (TypeError, ValueError):
            return "unknown", result
        if box.shape != (4,) or not np.isfinite(box).all():
            return "unknown", result
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        if x1 >= x2 or y1 >= y2:
            return "unknown", result
        x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        x2, y2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        if x1 >= x2 or y1 >= y2:
            return "unknown", result
        color = self.detect_color(image[y1:y2, x1:x2])
        ink = {
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "unknown": (255, 255, 255),
        }[color]
        cv2.rectangle(result, (x1, y1), (x2 - 1, y2 - 1), ink, 2)
        cv2.putText(
            result, color.upper(), (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ink, 1
        )
        return color, result

    def is_red_yellow_combination(self, image, traffic_light_boxes):
        colors = {self.analyze_traffic_light(image, box)[0] for box in traffic_light_boxes}
        return {"red", "yellow"}.issubset(colors)
