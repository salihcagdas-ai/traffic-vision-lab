"""Ground-plane displacement over source-video time, never processing time."""

import numpy as np

from .homography_utils import HomographyTransformer, finite_points, positive_number


class SpeedEstimator(HomographyTransformer):
    """Estimate average displacement speed using one calibrated transformation.

    target_width and target_height describe the ground rectangle in metres
    when meter_per_pixel=1.0 (the default). The legacy parameter's actual meaning
    is metres per target coordinate unit, not metres per original image pixel.
    Pass raw image points; do not transform them before calculate_speed.
    """

    def __init__(self, source_points, target_width, target_height, fps, meter_per_pixel=1.0):
        super().__init__(source_points, target_width, target_height)
        self.fps = positive_number(fps, "fps")
        self.meter_per_pixel = positive_number(meter_per_pixel, "meter_per_pixel")

    def calculate_speed(self, points, elapsed_frames):
        """Return endpoint displacement speed in km/h.

        elapsed_frames is last source frame index minus first, so missing
        detections do not shorten elapsed time. It must be positive.
        Fewer than two samples return zero. This endpoint estimate suits a
        short straight-road window; it does not measure curved path length.
        """
        elapsed = positive_number(elapsed_frames, "elapsed_frames")
        pixels = finite_points(points)
        if len(pixels) < 2:
            return 0.0
        ground = self.transform_points(pixels[[0, -1]])
        metres = float(np.linalg.norm(ground[1] - ground[0])) * self.meter_per_pixel
        speed = metres / (elapsed / self.fps) * 3.6
        if not np.isfinite(speed):
            raise ValueError("speed calculation produced a non-finite value")
        return speed
