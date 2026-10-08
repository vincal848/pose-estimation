# Running-Form Analysis from Phone Video

[![tests](https://github.com/vincal848/pose-estimation/actions/workflows/tests.yml/badge.svg)](https://github.com/vincal848/pose-estimation/actions/workflows/tests.yml)

I want to point a phone at myself on a treadmill or a track and get real
gait numbers out of it -- cadence, ground contact time, vertical
oscillation, overstride, knee flexion, trunk lean -- instead of trusting
whatever a running watch's accelerometer infers. A side-view video and a
pose estimator can see the thing directly: where the heel is, where the
hip is, frame by frame. The watch has to guess all of that from a wrist
accelerometer.

Today the metric math is tested on synthetic landmark sequences with an
exact, known answer, and the video path (video -> landmarks -> report) is
built and tested end to end with a stand-in pose detector. What has *not*
happened: no real footage and no real MediaPipe model have been run
through it yet, so none of the numbers below describe real runners.

## Method

```
video -> landmarks per frame -> smoothing -> gait event detection -> metrics
```

1. **Video -> landmarks.** MediaPipe Pose (BlazePose), 33 keypoints per
   frame, run on a side-view clip. Output: an `(frames, 33, 3)` array of
   `(x_px, y_px, visibility)`; frames with no detection are NaN with
   visibility 0. Built in `video.py` (M2), but only the plumbing is
   exercised by tests -- the MediaPipe call itself has not been run.
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

- **Cadence vs. a hand count** of foot strikes in a public clip (no
  watch reference for public footage). Target: within 2 spm.
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
- [x] **M2** -- MediaPipe wrapper (`video.py`) + landmark export to disk.
      Code and plumbing tests done; the MediaPipe call is unrun until a
      model file and a real clip are in hand.
- [ ] **M3** -- Event detection validated on real video, not just synthetic
      data.
- [ ] **M4** -- Cadence validated against a hand-counted reference on a
      public clip (no GPS watch reading available).
- [ ] **M5** -- A per-run report: the metrics above, with the figures that
      back them up.

## Success metrics

- Cadence within 2 spm of a hand count on the public clip.
- Contact-time repeatability (coefficient of variation across trials at
  the same pace) tight enough to tell two different running form changes
  apart, not just tight in the absolute.

## Status

M1 done. M2 code done, MediaPipe call itself unverified. The pipeline
`cli.py` (video -> landmarks -> report) runs end to end in tests on
synthetic landmarks, including dropped and low-confidence frames. M3 and M4
need a real clip; the plan is an openly licensed public video, so there is
no smartwatch cadence comparison -- M4 becomes "cadence agrees with a hand
count of strikes in the clip."

## Running on a video

```bash
pip install -r requirements-dev.txt -r requirements-video.txt
# current mediapipe only has the Tasks API: it needs a pose_landmarker
# .task model file (Google publishes pose_landmarker_{lite,full,heavy}).
# It is not bundled or downloaded by this repo; keep it outside git.
python cli.py clip.mp4 --model pose_landmarker_full.task     --save-landmarks clip.npz --m-per-px 0.002 --facing right
python cli.py clip.npz        # re-analyse saved landmarks, no model needed
```

The report gives cadence (steps/min, both feet, median step interval),
ground contact time, vertical oscillation, overstride, trunk lean and knee
flexion at strike and peak. Pixel values are always reported; metre values
only if `--m-per-px` is given. Short detector dropouts (<= 0.2 s) are
interpolated; events touching longer gaps are discarded. Videos stay out of
the repo (`data/`, `*.mp4` are git-ignored).

## Quick start

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
pytest tests -q   # 61 tests
```

## Repository guide

| Path | Contents |
|---|---|
| `landmarks.py` | MediaPipe's 33-point layout, joint lookup, pixel-to-metre scale |
| `kinematics.py` | Smoothing, gait event detection, every gait metric |
| `synth.py` | Synthetic landmark arrays with known, exact gait events |
| `video.py` | Frame reading, detector seam, MediaPipe PoseLandmarker adapter, `.npz` export |
| `analyze.py` | Pure landmarks -> metrics report; gap interpolation, low-confidence handling |
| `cli.py` | `python cli.py video-or-npz` -> printed report |
| `tests/` | 61 tests, synthetic data only, no network, no model downloads |
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
