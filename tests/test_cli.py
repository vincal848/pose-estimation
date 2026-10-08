"""End to end: video file -> report, with a stand-in pose detector."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import cli
import video
from synth import make_synthetic_gait


def _cadence(out: str) -> float:
    return float(next(l for l in out.splitlines() if l.startswith("cadence_spm")).split()[-1])


def test_video_to_report_recovers_the_generators_cadence(tmp_path, capsys):
    fps = 60
    lm, _ = make_synthetic_gait(cadence_spm=165.0, contact_time_s=0.24, fps=fps, seconds=10, noise=1.0, seed=2)
    path = str(tmp_path / "blank.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (32, 32))
    for _ in lm:
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()

    it = iter(lm)
    assert cli.main([path, "--save-landmarks", str(tmp_path / "lm.npz")], detect=lambda _f: [next(it)]) == 0
    assert _cadence(capsys.readouterr().out) == pytest.approx(165.0, abs=2.0)

    saved, saved_fps = video.load_landmarks(str(tmp_path / "lm.npz"))
    assert saved.shape == lm.shape and saved_fps == pytest.approx(fps)
    cli.main([str(tmp_path / "lm.npz")])
    assert _cadence(capsys.readouterr().out) == pytest.approx(165.0, abs=2.0)
