"""Pick the runner out of a multi-person video. Pure numpy.

Poses are linked frame to frame by nearest hip into tracks. The runner is
the track whose ankles oscillate periodically over the first seed_s seconds
(a bystander's do not), or, with an roi, the first track seen with its hip
inside the box.
"""

import numpy as np
from numpy.typing import NDArray

from landmarks import NUM_LANDMARKS, POSE_LANDMARKS as P

Pose = NDArray[np.float64]  # (33, 3): x_px, y_px, visibility


def _hip(pose: Pose) -> NDArray[np.float64]:
    return pose[[P["left_hip"], P["right_hip"]], :2].mean(axis=0)


def _torso(pose: Pose) -> float:
    shoulder = pose[[P["left_shoulder"], P["right_shoulder"]], :2].mean(axis=0)
    return float(np.linalg.norm(shoulder - _hip(pose))) or 1.0


def build_tracks(
    poses: list[list[Pose]], max_gap: int, jump_torsos: float = 1.5
) -> list[dict[int, Pose]]:
    """Greedy nearest-hip linking. A pose joins a track whose last hip is
    within jump_torsos torso lengths and was seen within max_gap frames;
    otherwise it starts a new track. ponytail: greedy, so two people
    crossing paths can swap tracks; use Hungarian matching if that bites."""
    tracks: list[dict[int, Pose]] = []
    last: list[int] = []
    for i, frame in enumerate(poses):
        taken: set[int] = set()
        for pose in frame:
            best, best_d = None, jump_torsos * _torso(pose)
            for t, tr in enumerate(tracks):
                if t in taken or i - last[t] > max_gap:
                    continue
                d = float(np.linalg.norm(_hip(tr[last[t]]) - _hip(pose)))
                if d <= best_d:
                    best, best_d = t, d
            if best is None:
                tracks.append({i: pose})
                last.append(i)
                taken.add(len(tracks) - 1)
            else:
                tracks[best][i] = pose
                last[best] = i
                taken.add(best)
    return tracks


def _ankle_score(track: dict[int, Pose], n_seed: int, fps: float) -> float:
    """Ankle swing (std of ankle y relative to hip, in torsos) times the
    share of its spectrum in 1-4 Hz. 0 if the track covers < half the seed."""
    frames = [i for i in track if i < n_seed]
    if len(frames) < max(n_seed // 2, 8):
        return 0.0
    ys = np.array([[track[i][P[f"{s}_ankle"], 1] - _hip(track[i])[1] for s in ("left", "right")]
                   for i in frames]) / np.median([_torso(track[i]) for i in frames])
    best = 0.0
    for col in ys.T:
        col = col - col.mean()
        power = np.abs(np.fft.rfft(col)) ** 2
        freq = np.fft.rfftfreq(len(col), 1.0 / fps)  # ponytail: assumes gap-free frames
        total = power[1:].sum()
        if total > 0:
            best = max(best, float(col.std() * power[(freq >= 1) & (freq <= 4)].sum() / total))
    return best


def select_runner(
    poses: list[list[Pose]],
    fps: float,
    roi: tuple[float, float, float, float] | None = None,
    seed_s: float = 2.0,
) -> NDArray[np.float64]:
    """(frames, 33, 3) landmarks of the runner; frames without them are NaN
    with visibility 0. poses[i] is every pose detected in frame i."""
    tracks = build_tracks(poses, max_gap=int(round(0.5 * fps)))
    if roi is None:
        scores = [_ankle_score(t, int(round(seed_s * fps)), fps) for t in tracks]
        if not scores or max(scores) <= 0:
            raise ValueError("no person with periodic ankle motion found; pass --roi")
        chosen = tracks[int(np.argmax(scores))]
    else:
        x0, y0, x1, y1 = roi
        inside = [min((i for i, p in t.items() if x0 <= _hip(p)[0] <= x1 and y0 <= _hip(p)[1] <= y1), default=None)
                  for t in tracks]
        firsts = [(i, k) for k, i in enumerate(inside) if i is not None]
        if not firsts:
            raise ValueError(f"no person's hip falls inside roi {roi}")
        chosen = tracks[min(firsts)[1]]
    out = np.full((len(poses), NUM_LANDMARKS, 3), np.nan)
    out[:, :, 2] = 0.0
    for i, p in chosen.items():
        out[i] = p
    return out
