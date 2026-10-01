"""video.py stays importable, and its stub, without mediapipe installed."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

import video


def test_extract_landmarks_raises_not_implemented_or_import_error():
    # mediapipe is optional (requirements-video.txt). Either it is missing,
    # in which case the lazy import inside extract_landmarks raises
    # ImportError, or it is present and the M2 stub raises
    # NotImplementedError -- both mean "cannot extract yet," which is the
    # one thing this test is pinning down.
    with pytest.raises((NotImplementedError, ImportError)):
        video.extract_landmarks("nonexistent.mp4")
