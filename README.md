# Running-Form Analysis from Phone Video

[![tests](https://github.com/vincal848/pose-estimation/actions/workflows/tests.yml/badge.svg)](https://github.com/vincal848/pose-estimation/actions/workflows/tests.yml)

I want to point a phone at myself on a treadmill or a track and get real
gait numbers out of it -- cadence, ground contact time, vertical
oscillation, overstride, knee flexion, trunk lean -- instead of trusting
whatever a running watch's accelerometer infers. A side-view video and a
pose estimator can see the thing directly: where the heel is, where the
hip is, frame by frame. The watch has to guess all of that from a wrist
accelerometer.

This is the "ideas + scaffold" stage. There is no video pipeline yet. What
exists is the part I did not want to get wrong by building it against real
footage first: the event-detection and metric math, tested against
synthetic landmark sequences with an exact, known answer. Everything about
getting from an .mp4 to a landmark array is designed and documented, not
built.

## Method

```
video -> landmarks per frame -> smoothing -> gait event detection -> metrics
```

1. **Video -> landmarks.** MediaPipe Pose (BlazePose), 33 keypoints per
   frame, run on a side-view clip. Output: an `(frames, 33, 3)` array of
   `(x_px, y_px, visibility)`. Not built yet -- `video.py` documents the
   intended call and raises until M2.
2. **Smoothing.** Savitzky-Golay or a One Euro Filter, both parameterized
   in seconds rather than frames so they scale with the video's fps. The
   One Euro Filter's adaptive cutoff matters for event timing specifically:
   a fixed-cutoff low pass that is gentle enough to look good on a hip
   trace will also blur the fast part of a heel trace right at a strike.
3. **Gait event detection.** A foot strike is the start of a ground-contact
   phase: the heel is near its lowest point on screen (largest pixel `y`,
   since image `y` grows downward) and nearly stationary. Toe-off is the
   end of that phase. See `docs/DESIGN.md` for the exact thresholds.
4. **Metrics**, defined precisely in `docs/DESIGN.md`: cadence (steps/min,
   from strike timing), ground contact time (toe-off minus strike),
   vertical oscillation (half the hip's peak-to-trough bounce), overstride
   (heel-ahead-of-hip distance at strike), knee flexion (hip-knee-ankle
   angle, at strike and at its peak during swing), trunk lean (shoulder
   angle off vertical from the hip).

**Pixel-to-metre scale.** Every pixel measurement is converted to metres
exactly once, through one known real-world length -- the runner's
standing height, or a marker of known size in the frame -- via
`landmarks.pixel_scale`. No metric trusts camera distance or lens
assumptions instead.

**Frame rate.** 60 fps is the floor for contact-time work; at 60 fps a
frame is already 16.7 ms, which is a meaningful fraction of a ~200 ms
contact phase. 240 fps (most phones can do this in slow-motion mode) gets
that down to 4.2 ms and is what I would actually trust for contact time.
Cadence is far more forgiving, since it comes from timing between strikes
seconds apart, not from one short phase's duration.

## Validation

Planned, once there is real video (M3/M4):

- **Cadence vs. a GPS watch**, on the same run. Target: within 2 spm.
- **Contact time vs. published ranges** for recreational and trained
  runners, and against a treadmill with pressure sensing if I can get
  access to one.
- **Inter-trial repeatability**: same runner, same pace, multiple trials,
  checking the spread in each metric rather than just a single number.

What exists today is the synthetic-data equivalent -- recovering a known,
exact answer from landmark arrays built to have one. From the current test
suite, run on a 10-second, 170 spm, 250 ms contact-time synthetic clip:

| fps | cadence recovered | cadence error | mean contact time | contact time error | one frame |
|---|---|---|---|---|---|
| 60 | 169.634 spm | 0.366 spm | 0.2369 s | 0.01667 s | 0.01667 s |
| 240 | 169.930 spm | 0.070 spm | 0.2467 s | 0.00417 s | 0.00417 s |

Contact-time error is bounded by exactly one frame period at both rates,
which is the property the test asserts rather than the specific numbers
above (those will shift slightly with any threshold tuning).

## Limitations

- **2D, single side view.** No out-of-plane motion, no frontal-plane
  metrics (no pelvic drop, no knee valgus). A camera that is not level, or
  not perpendicular to the direction of travel, biases every pixel
  measurement in a way this project does not correct for.
- **Camera angle and distance** change the effective pixel scale across
  the frame if the runner moves toward or away from the camera (true on a
  track, mostly false on a treadmill, where the runner stays roughly in
  place).
- **Loose clothing** hides the true joint position under fabric; MediaPipe
  is trained on visible joints and will do something, not necessarily the
  right thing, when they are not.
- **Landmark jitter** is real even on clean footage. Smoothing helps, but a
  heavy hand on smoothing is exactly what risks shifting event timing,
  which is why `smooth()`'s effect on strike timing has its own test.
- **30 fps video gives ~33 ms time resolution.** Said plainly: a contact
  phase is roughly 150-250 ms, so at 30 fps you are resolving it to within
  one-seventh of its own duration at best. That is usable for cadence, not
  for a contact-time number I would trust.

## Milestones

- [x] **M1** -- Kinematics on synthetic landmark arrays. This scaffold:
      `kinematics.py`, `landmarks.py`, `synth.py`, tested without any
      video or model.
- [ ] **M2** -- MediaPipe wrapper (`video.py`) + landmark export to disk.
- [ ] **M3** -- Event detection validated on real video, not just synthetic
      data.
- [ ] **M4** -- Cadence validated against a GPS watch's own reading, on N
      real runs.
- [ ] **M5** -- A per-run report: the metrics above, with the figures that
      back them up.

## Success metrics

- Cadence within 2 spm of a GPS watch's own reading, across runs.
- Contact-time repeatability (coefficient of variation across trials at
  the same pace) tight enough to tell two different running form changes
  apart, not just tight in the absolute.

## Status

Scaffold. M1 in progress.

## Quick start

```bash
pip install -r requirements.txt
```

```python
from synth import make_synthetic_gait
from landmarks import joint_xy
import kinematics as k

landmarks, truth = make_synthetic_gait(
    cadence_spm=170, contact_time_s=0.25, fps=60, seconds=10, seed=0)

heel_y = joint_xy(landmarks, "left_heel")[:, 1]
strikes = k.detect_foot_strikes(heel_y, fps=60)
k.cadence_spm(strikes) * 2   # both-feet cadence from one tracked foot
```

```bash
pytest tests -q   # 39 tests
```

## Repository guide

| Path | Contents |
|---|---|
| `landmarks.py` | MediaPipe's 33-point layout, joint lookup, pixel-to-metre scale |
| `kinematics.py` | Smoothing, gait event detection, every gait metric |
| `synth.py` | Synthetic landmark arrays with known, exact gait events |
| `video.py` | MediaPipe extraction -- stubbed, API documented, lands in M2 |
| `tests/` | 39 tests, synthetic data only, no network, no model downloads |
| `docs/DESIGN.md` | Landmark indices used, event-detection logic, metric formulas |

## Future interests

- Both-feet tracking in one pass, instead of running strike detection
  twice and merging -- matters once the near and far leg are not
  symmetric in the frame.
- A report generator (M5) that lays out the metrics and the raw traces
  they came from on one page per run, so a bad detection is visible, not
  just a wrong number.
- Comparing contact time and vertical oscillation against a force plate
  or pressure-sensing treadmill, which is the only way to validate those
  two against something better than "plausible."

## Notes

- No video files are committed to this repo (`.gitignore` excludes
  `data/`, `*.mp4`, `*.mov`), and nothing in the test suite downloads a
  model or touches the network.
- `mediapipe` and `opencv-python` are optional (`requirements-video.txt`)
  and imported lazily inside `video.py`; the rest of the package never
  needs them.
