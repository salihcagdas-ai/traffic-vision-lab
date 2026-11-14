"""Calibrated image-to-ground transforms derived from the shared source project."""

import cv2
import numpy as np


def finite_points(points, *, name="points"):
    """Return finite float64 coordinate pairs, including empty input."""
    try:
        values = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric coordinate pairs") from exc
    if values.shape == (0,):
        values = values.reshape(0, 2)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must have shape (N, 2) and finite coordinates")
    return values


def positive_number(value, name):
    """Validate a positive finite scalar, without accepting Boolean values."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


class HomographyTransformer:
    """Map consecutive convex image corners to a measured ground rectangle.

    Corners correspond to (0, 0), (width, 0), (width, height), (0, height).
    Width and height are physical dimensions, normally metres, not raster
    dimensions. Image rendering has a separate pixels_per_unit scale.
    """

    def __init__(self, source_points, target_width, target_height):
        source = finite_points(source_points, name="source_points")
        if source.shape != (4, 2):
            raise ValueError("source_points must contain exactly four corners")
        edges = np.roll(source, -1, axis=0) - source
        next_edges = np.roll(edges, -1, axis=0)
        turns = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
        tolerance = max(float(np.max(np.abs(edges))) ** 2, 1.0) * 1e-12
        if not (np.all(turns > tolerance) or np.all(turns < -tolerance)):
            raise ValueError("source_points must be an ordered, convex quadrilateral")
        self.target_width = positive_number(target_width, "target_width")
        self.target_height = positive_number(target_height, "target_height")
        self.source_points = source.copy()
        self.target_points = np.array(
            [
                [0, 0],
                [self.target_width, 0],
                [self.target_width, self.target_height],
                [0, self.target_height],
            ],
            dtype=np.float64,
        )
        with np.errstate(over="ignore"):
            source32 = self.source_points.astype(np.float32)
            target32 = self.target_points.astype(np.float32)
        if not np.isfinite(source32).all() or not np.isfinite(target32).all():
            raise ValueError("calibration coordinates exceed the supported range")
        self.transform_matrix = cv2.getPerspectiveTransform(source32, target32)
        if (
            not np.isfinite(self.transform_matrix).all()
            or np.linalg.matrix_rank(self.transform_matrix) < 3
        ):
            raise ValueError("calibration does not define an invertible homography")
        self.inverse_transform_matrix = np.linalg.inv(self.transform_matrix)

    @staticmethod
    def _transform(points, matrix):
        values = finite_points(points)
        if not len(values):
            return np.empty((0, 2), dtype=np.float64)
        projected = np.column_stack((values, np.ones(len(values)))) @ matrix.T
        divisor = projected[:, 2]
        if np.any(np.abs(divisor) <= 1e-12):
            raise ValueError("a point lies on the homography horizon")
        transformed = projected[:, :2] / divisor[:, None]
        if not np.isfinite(transformed).all():
            raise ValueError("projection produced a non-finite coordinate")
        return transformed

    def transform_point(self, point):
        return self.transform_points([point])[0]

    def transform_points(self, points):
        """Transform image pixels to ground coordinates without rounding."""
        return self._transform(points, self.transform_matrix)

    def inverse_transform_point(self, point):
        return self.inverse_transform_points([point])[0]

    def inverse_transform_points(self, points):
        return self._transform(points, self.inverse_transform_matrix)

    @staticmethod
    def _image(image):
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or not image.size:
            raise ValueError("image must be a non-empty image array")
        return image

    def _render_transform(self, pixels_per_unit):
        scale = positive_number(pixels_per_unit, "pixels_per_unit")
        width = int(np.ceil(self.target_width * scale)) + 1
        height = int(np.ceil(self.target_height * scale)) + 1
        if width > 32000 or height > 32000 or width * height > 50_000_000:
            raise ValueError("requested bird's-eye image is too large")
        matrix = np.diag([scale, scale, 1.0]) @ self.transform_matrix
        return matrix, (width, height)

    def transform_image(self, image, pixels_per_unit=10.0):
        """Render a raster, by default at ten pixels per ground unit."""
        matrix, size = self._render_transform(pixels_per_unit)
        return cv2.warpPerspective(self._image(image), matrix, size)

    def inverse_transform_image(self, bird_view_image, original_shape, pixels_per_unit=10.0):
        """Invert a raster rendered with the same pixels_per_unit value."""
        if len(original_shape) < 2:
            raise ValueError("original_shape must provide height and width")
        height, width = original_shape[:2]
        if int(height) != height or int(width) != width or height <= 0 or width <= 0:
            raise ValueError("original_shape must contain positive integer dimensions")
        matrix, _ = self._render_transform(pixels_per_unit)
        return cv2.warpPerspective(
            self._image(bird_view_image), np.linalg.inv(matrix), (int(width), int(height))
        )

    def draw_region_on_image(self, image, color=(0, 255, 0), thickness=2):
        result = self._image(image).copy()
        points = np.rint(self.source_points).astype(np.int32).reshape(-1, 1, 2)
        return cv2.polylines(result, [points], True, color, thickness)

    def visualize_transformation(self, image, pixels_per_unit=10.0):
        marked = self.draw_region_on_image(image)
        bird_view = self.transform_image(image, pixels_per_unit)
        height = marked.shape[0]
        width = max(1, round(bird_view.shape[1] * height / bird_view.shape[0]))
        return np.hstack((marked, cv2.resize(bird_view, (width, height))))
