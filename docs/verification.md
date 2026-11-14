# Verification notes

The base environment is Python 3.12 on Windows, CPU, with the versions recorded
in `requirements.txt` and `requirements-dev.txt`.

On 15 November 2026, **79 tests passed**, including the optional backend smoke
test with Ultralytics 8.4.213, PyTorch 2.15.0 CPU and LAP 0.5.14. Ruff and dependency
consistency checks passed. The base suite also works without model packages;
the optional backend smoke test is skipped in that environment.

## What the automated checks establish

- A known image displacement is mapped once into the measured plane, then
  divided by the elapsed frame interval. Non-finite and degenerate calibration
  inputs are rejected.
- Directed crossings are constrained to the finite stop line. Approaching,
  waiting, crossing the line's extension and travelling in the reverse direction
  do not produce the same event as a forward crossing on red.
- Empty or ambiguous signal regions are reported as unknown.
- OCR token parsing accepts numeric confidence values and joins complete lines.
  Unit tests inject OCR results; they do not validate recognition from real plates.
- SQLite parameter binding, foreign keys and default personal-field suppression
  are checked. Each CLI run uses a fresh output directory and database.
- The deterministic demo writes a readable 90-frame MP4 and the expected event
  summary. The video runner's integration is checked with a synthetic video and
  an injected detector, including dimension mismatch and missing input failures.
- The installed Ultralytics/ByteTrack adapter loads locally generated, randomly
  initialized weights and processes two blank frames while socket connections
  are blocked. This verifies local model loading and API compatibility, not
  pretrained vehicle detection or tracking accuracy. The optional plate crop
  path is tested with injected OCR; no actual Tesseract executable was run.

## What is not established

No original trained detector, plate model, labelled road dataset or recorded
camera calibration is bundled. Real detection/tracking quality, OCR accuracy,
speed-estimation error and processing FPS remain unmeasured. Synthetic results
are not substitutes for a labelled evaluation set.

The original GUI scripts and hand-picked footage are not part of this edition.
The current CLI has a single monitored plane/signal and requires constant-rate
video. Homography assumes a flat road and a stationary camera. Bounding-box
bottom centres only approximate the ground-contact point. Track ID switches,
occlusion, turning, camera motion, shadows and inaccurate dimensions can all
affect event results. Average displacement is not travelled path length on curves.

HSV signal classification is a simple color heuristic without temporal smoothing.
Unknown/yellow states emit no red-light event. The stop-line frame convention is
documented in the README; it is not a statement of traffic-law interpretation.

## Upstream references

- [Original project and source revision](../NOTICE.md)
- [Ultralytics tracking API and ByteTrack selection](https://docs.ultralytics.com/modes/track/)
- [Ultralytics configuration](https://docs.ultralytics.com/usage/cfg/)
- [PyTesseract source and usage](https://github.com/madmaze/pytesseract)

The source project's headline performance claims are not carried into this
edition because their datasets, weights and evaluation procedure are unavailable.
