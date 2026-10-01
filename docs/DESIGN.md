# Design

## Landmark array

Every function in this repo that touches a full pose, rather than a single
joint's series, agrees on one shape: `(frames, 33, 3)`, channels
`(x_px, y_px, visibility)`. That is MediaPipe Pose's 33-point BlazePose
topology, in image-pixel space -- not MediaPipe's own normalized `[0, 1]`
or world-coordinate output. Converting to metres happens exactly once, via
`landmarks.pixel_scale`, from one known real-world length (the runner's
height, or a marker of known size in the scene). Image `y` grows downward,
the usual convention for image arrays; every place that matters (gait
events, vertical oscillation) says so again at the point of use, because it
is the easiest sign to get backwards.

### The 33 indices used here

| # | Name | Used by |
|---|---|---|
| 11 | left_shoulder | trunk_lean |
| 12 | right_shoulder | trunk_lean |
| 23 | left_hip | vertical_oscillation, overstride, trunk_lean, joint_angle (hip) |
| 24 | right_hip | same, right side |
| 25 | left_knee | joint_angle (knee flexion) |
| 26 | right_knee | joint_angle, right side |
| 27 | left_ankle | joint_angle (knee flexion, ankle end) |
| 28 | right_ankle | joint_angle, right side |
| 29 | left_heel | detect_foot_strikes, detect_toe_offs, overstride |
| 30 | right_heel | same, right side |

The other 23 points (face, arms, hands, foot index) are part of the
MediaPipe topology and are carried through `landmarks.py`'s general
indexing, but no metric in this project reads them yet. See
`landmarks.POSE_LANDMARKS` for the full name-to-index table.

## Event detection

A foot strike and its matching toe-off bound one ground-contact phase:
the heel is close to its largest `y` (nearest the ground, since `y` grows
downward) and nearly stationary. `kinematics._contact_runs` finds this
directly, without a model of the gait cycle:

1. Vertical velocity `vy = d(heel_y)/dt` by finite difference.
2. "Near ground": `heel_y` within the top `ground_frac` (default 25%) of
   its own 5th-95th percentile range. Percentiles rather than min/max so a
   single noisy frame cannot move the threshold.
3. "Still": `|vy|` below `vel_frac` (default 50%) of the clip's velocity
   standard deviation.
4. A contact phase is a run of frames meeting both conditions,
   `min_run_frames` (default 2) or longer. Its first frame is the strike;
   one past its last frame is the toe-off.
5. A run still open at the last frame of the clip is dropped. We never see
   it close, so there is no way to tell a genuine, completed ground
   contact from a swing that happened to be near the ground when the
   recording cut off.

This is a threshold detector, not a learned or model-based one, which is
the right amount of machinery for a side-view heel trace with a clearly
bimodal (ground / swing) vertical position. It is also exactly what
`synth.py`'s flat-contact, arced-swing heel trajectory was built to make
unambiguous, so that recovery error in the tests measures the detector,
not the data.

`cadence_spm` just counts strikes over the time between the first and
last one. A single tracked foot's own strike rate is half of what a GPS
watch reports (full, both-feet cadence) -- either double it, or detect
strikes on both heels and merge the two time series before counting, which
is what the cadence test in `tests/test_kinematics.py` does.

## Metric definitions

- **Ground contact time** = `toe_off_time - strike_time`, per strike.
  `ground_contact_times(strikes, toe_offs)` assumes the two arrays are
  already paired index-for-index, which is what calling
  `detect_foot_strikes` and `detect_toe_offs` on the same `heel_y` gives
  you.
- **Cadence (steps/min)** = `60 * (n - 1) / (t[-1] - t[0])` for `n` strike
  times `t`. See the single-foot-vs-both-feet note above.
- **Vertical oscillation** = `(max(hip_y) - min(hip_y)) / 2`, converted
  from pixels to metres. Half the peak-to-trough range, matching how
  running watches report this number as an amplitude rather than a full
  excursion.
- **Overstride** = `(heel_x - hip_x)` at a strike's frame index, converted
  to metres. Positive means the heel landed ahead of the hip in the
  direction of increasing `x`; whether that direction is the runner's
  direction of travel is a convention the caller has to get right for
  their camera setup, which is why the function does not guess it.
- **Trunk lean** = `atan2(shoulder_x - hip_x, -(shoulder_y - hip_y))` in
  degrees, per frame. Zero when the shoulder sits directly above the hip;
  the `-(...)` accounts for `y` growing downward, so "up" is negative `y`.
- **Joint angle** (e.g. knee flexion) = the angle at vertex `b` of
  `a-b-c`, from the dot product of `b->a` and `b->c`, clipped before
  `arccos` to absorb floating-point drift just past +-1.

## Smoothing

Two interchangeable methods, both taking `fps` so their parameters are in
seconds rather than frames:

- **Savitzky-Golay** (`scipy.signal.savgol_filter`): a fixed-length
  polynomial fit over a sliding window, `window_s` seconds wide. Good
  default when you want a smooth curve for display or for angle metrics
  that do not depend on exact event timing.
- **One Euro Filter** (Casiez, Pietriga & Roussel, 2012): a causal,
  adaptive-cutoff low-pass -- the cutoff rises with the signal's own speed,
  so it smooths a slow-moving hip trace hard while barely touching the
  fast part of a heel trace right at a strike. That matters here because
  strike/toe-off timing comes from exactly the fast part of the signal a
  fixed-cutoff filter would blur. `tests/test_kinematics.py` checks this
  property directly: smoothing must not move a detected strike by more
  than one frame period.
