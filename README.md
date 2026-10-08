# Running-Form Analysis from Phone Video

[![tests](https://github.com/vincal848/pose-estimation/actions/workflows/tests.yml/badge.svg)](https://github.com/vincal848/pose-estimation/actions/workflows/tests.yml)

I want to point a phone at myself on a treadmill or a track and get real
gait numbers out of it -- cadence, ground contact time, vertical
oscillation, overstride, knee flexion, trunk lean -- instead of trusting
whatever a running watch's accelerometer infers. A side-view video and a
pose estimator can see the thing directly: where the heel is, where the
hip is, frame by frame. The watch has to guess all of that from a wrist
accelerometer.

The metric math is tested on synthetic landmark sequences with an exact,
known answer, and the video path (video -> landmarks -> report) is tested
end to end with a stand-in pose detector and has been run once on a real
public treadmill clip (see "First real-video result"). The synthetic table
in Validation below is not real-runner data.

## Method

```
video -> landmarks per frame -> smoothing -> gait event detection -> metrics
```

1. **Video -> landmarks.** MediaPipe Pose (BlazePose), 33 keypoints per
   frame, run on a side-view clip. Output: an `(frames, 33, 3)` array of
   `(x_px, y_px, visibility)`; frames with no detection are NaN with
   visibility 0. Built in `video.py` (M2); tests cover the plumbing, and the
   MediaPipe call has run once on a real clip.
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

Real-video checks (M3/M4), first pass done on one clip:

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
      Run on one real clip.
- [~] **M3** -- Event detection checked on one real clip against a hand
      count of strikes on one clip (cadence agrees); a held-out clip (Arakawa river) was refused
      by the pipeline (runner too small for MediaPipe); contact time not validated.
- [~] **M4** -- Cadence agrees with a hand count on one public clip (163 vs
      163 spm); a single successful clip.
- [ ] **M5** -- A per-run report: the metrics above, with the figures that
      back them up.

## Success metrics

- Cadence within 2 spm of a hand count on the public clip.
- Contact-time repeatability (coefficient of variation across trials at
  the same pace) tight enough to tell two different running form changes
  apart, not just tight in the absolute.

## Status

M1 and M2 done; the MediaPipe call has run on one real clip. The pipeline
`cli.py` (video -> landmarks -> report) runs end to end in tests on
synthetic landmarks, including dropped and low-confidence frames. M3 and M4
are partial: one public clip, cadence checked against a hand count (no
smartwatch reference exists for public footage), contact time and
oscillation not independently validated.

## First real-video result

One clip, one run, parameters tuned on this same clip -- read it as a smoke test
of the pipeline, not a validation study. Video: Arellano et al. 2015, PLOS
ONE S1 video, treadmill side view at 3.0 m/s, CC BY 4.0 (see `CREDITS.md`).
854x480, **29.97 fps**, 299 frames (10 s); MediaPipe `pose_landmarker_full`.
Only the legs and lower hips are in frame, and a foreground object hides
the far leg for much of the clip.

| metric | pipeline | hand check |
|---|---|---|
| cadence | 162.9 spm (whole clip) | 15 landings by eye in frames 21-175, 14 steps in 154 frames = 163.4 spm |
| strikes, same window | 8 near-foot strikes (frames 23-177), x2 for both feet = 15 landings | 15 landings |
| ground contact time | 0.200 s (6 frames) | stance looks like ~8 frames (~0.27 s) by eye |
| vertical oscillation | 18.8 px (no metre scale: runner height unknown) | not checked |
| overstride / trunk lean | +9.7 px / 8.3 deg with `--facing left` | not checked |

Sanity: 3.0 m/s at 163 spm is a 1.10 m step, in the usual range for
recreational running.

### Three clips, three roles

| clip | role | pipeline cadence | hand cadence | pipeline contact | hand contact |
|---|---|---|---|---|---|
| s003, treadmill 3.0 m/s | tuning (`vel_frac`, cadence method chosen here) | 162.9 spm | 163.4 spm | 0.200 s | ~0.27 s |
| s005, treadmill 9.0 m/s | development (it motivated runner selection) | 240.8 spm | ~232 spm (landings at frames 44 to 90.5, 6 steps) | 0.100 s | ~2-3 frames, ~0.1 s |
| Arakawa river jog (Nesnad, CC BY 4.0), 1920x1080, 29.97 fps | held-out | **refused** (no person with periodic ankle motion) | ~185 spm (3 steps in frames 285-312, +-9 spm) | none | not measured |

How each was run and what went wrong:

- **s003** was used to pick the method and thresholds, so agreement there
  proves little.
- **s005** first produced no result, because MediaPipe followed a
  bystander. With multi-person detection and runner selection (`track.py`)
  the pipeline picks the sprinter (hip near x=364 px) and reports 240.8 spm
  against a hand count of about 232 spm: 9 spm (4%) high, outside the 2 spm
  target, though the hand count is only good to about +-5 spm. My first
  hand count for this clip (~210 spm) was wrong -- it used blurry coarse
  frames and a window that included the belt starting up -- and was redone
  at finer frames. Ground contact (0.100 s, 3 frames) cannot be told apart
  from the frame period at 30 fps. s005 is a development clip now; it has
  not been used to change any threshold.
- **Arakawa** was frozen at commit 088c1d7 before running. The runner is
  clearly visible in side view for only about a second (frames ~285-312),
  about 230 px tall in a 1080p panning shot. With default settings the
  pipeline refuses; a diagnostic run on that one second found zero poses
  in every frame, so MediaPipe's detector does not see a person this small.
  This is a failure of detection on a small subject, and the tracker's
  refusal is the correct output. No tuning was done on this clip. A usable
  held-out result therefore still does not exist; the next step is a
  crop/zoom to the runner before extraction, or a closer clip.
- Rejected as held-out candidates before running anything: `Running form.ogv`
  (legs below the knee only, no hips), a pole-vault athlete video, and a
  park-path video from a moving cyclist.

## Input requirements

The pipeline's documented operating conditions. They were written down
before searching for the next held-out clip and are not tuned to any
result; a clip outside them is not a test of the pipeline.

- Side view: camera roughly perpendicular to the running direction.
- Whole body visible, including both ankles, for the whole analysed stretch.
- Runner at least ~400 px tall in the frame.
- At least 4 s of steady running in frame.
- At least 25 fps (60+ for any contact-time claim).
- A single runner, or one clearly dominant (large, central) person.

**Held-out status.** No clip meeting these requirements was found, so there
is no held-out result and none is claimed. Searched (metadata, thumbnails
and first frames only; the pipeline was not run on any of them): Wikimedia
Commons search and the running/jogging video categories, and Internet
Archive search for openly licensed running video. Pexels and Pixabay could
not be searched (automated access is blocked by a bot check), so they are
unexamined, not ruled out. What turned up fails the requirements: the Arakawa jog
(runner far too small and visible ~1 s), `Running form.ogv` (legs below the
knee only), 320x240 coaching clips (runner far under 400 px), a CC0 Science
Nation news segment on gait retraining (cut compilation, 640x360, no 4 s
steady full-body side view), and the rest are scientific videos of animals
or other subjects. The remaining Wikimedia running videos are the paper's
own s003/s005 treadmill clips, already used for tuning and development.

Pexels 4065472 ("A man jogging in the street near the lake", 1920x1080,
29.97 fps, 12.8 s, Pexels licence; sha1 f60acd47b30cd45ef4b09fa43ade711404bff7d5)
was judged from twelve evenly spaced frames and rejected before any run:
it is a close tracking shot cropped at the thigh, so the knees and ankles are
out of frame, and lamp posts repeatedly occlude the runner. It meets the fps
and duration requirements but not "whole body including ankles". The two
other Pexels candidates found were 24 fps and fail the fps requirement.

A
real held-out test needs a clip contributed or filmed to the requirements
above (a phone clip of a runner at 60 fps from the side would do).

## Running on a video

```bash
pip install -r requirements-dev.txt -r requirements-video.txt
# current mediapipe only has the Tasks API: it needs a pose_landmarker
# .task model file (Google publishes pose_landmarker_{lite,full,heavy}).
# python scripts/fetch_video.py fetches it (and the clip below) into data/,
# checksum-verified; nothing is committed.
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
pytest tests -q   # 69 tests
```

## Repository guide

| Path | Contents |
|---|---|
| `landmarks.py` | MediaPipe's 33-point layout, joint lookup, pixel-to-metre scale |
| `kinematics.py` | Smoothing, gait event detection, every gait metric |
| `synth.py` | Synthetic landmark arrays with known, exact gait events |
| `scripts/fetch_video.py` | Checksum-verified download of the clip and model into `data/`; licences in `CREDITS.md` |
| `video.py` | Frame reading, detector seam, MediaPipe PoseLandmarker adapter, `.npz` export |
| `track.py` | Pure runner selection from multi-person detections (nearest-hip tracks, ankle periodicity or ROI) |
| `analyze.py` | Pure landmarks -> metrics report; gap interpolation, low-confidence handling |
| `cli.py` | `python cli.py video-or-npz` -> printed report |
| `tests/` | 69 tests, synthetic data only, no network, no model downloads |
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
