"""
MediaPipe bridge imported by MATLAB.

Exports:
    initialize(camera_index=0)
    capture_frame()
    detect_landmarks(frame_bytes, height, width)
    release()
"""

from __future__ import annotations

import sys
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


if not hasattr(mp, "solutions"):
    raise RuntimeError(
        "mediapipe has no attribute 'solutions' "
        f"(file={getattr(mp, '__file__', '<unknown>')}, "
        f"version={getattr(mp, '__version__', '<unknown>')})"
    )


_FRAME_WIDTH = 640
_FRAME_HEIGHT = 480

_cap = None
_face_mesh = None
_face_detector = None


def _open_capture(camera_index: int):
    """Open the webcam, preferring V4L2 on Linux, and validate with one read."""
    backends = []
    if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
        backends.append(cv2.CAP_V4L2)
    backends.append(None)

    for backend in backends:
        cap = (
            cv2.VideoCapture(int(camera_index), backend)
            if backend is not None
            else cv2.VideoCapture(int(camera_index))
        )
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, _FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _FRAME_HEIGHT)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
        if cap is not None:
            cap.release()

    return None


def initialize(camera_index: int = 0) -> bool:
    """Create FaceMesh and open the webcam without starting any preview loop."""
    global _cap, _face_mesh, _face_detector

    release()

    try:
        _face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.55,
        )
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        _cap = _open_capture(int(camera_index))
        if _cap is None:
            release()
            return False

        return True
    except Exception:
        release()
        return False


def capture_frame() -> Tuple[bytes, int, int]:
    """Capture one frame and return raw RGB bytes plus height and width."""
    if _cap is None or not _cap.isOpened():
        return b"", 0, 0

    ok, frame_bgr = _cap.read()
    if not ok or frame_bgr is None:
        return b"", 0, 0

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]
    return frame_rgb.tobytes(), int(height), int(width)


def detect_landmarks(frame_bytes, height: int, width: int) -> List[float]:
    """Rebuild an RGB frame from bytes and return flattened landmarks."""
    if _face_mesh is None or _face_detector is None:
        return []

    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        return []

    frame_np = np.frombuffer(bytes(frame_bytes), dtype=np.uint8)
    expected_size = height * width * 3
    if frame_np.size != expected_size:
        return []

    frame_rgb = frame_np.reshape((height, width, 3))
    det_results = _face_detector.process(frame_rgb)
    if not det_results.detections:
        return []

    score = float(det_results.detections[0].score[0])
    if score < 0.55:
        return []

    results = _face_mesh.process(frame_rgb)
    if not results.multi_face_landmarks:
        return []

    face_landmarks = results.multi_face_landmarks[0].landmark
    flat = []
    for lm in face_landmarks:
        flat.extend((float(lm.x), float(lm.y), float(lm.z)))
    return flat


def release() -> None:
    """Release camera and FaceMesh resources."""
    global _cap, _face_mesh, _face_detector

    if _cap is not None:
        try:
            _cap.release()
        finally:
            _cap = None

    if _face_mesh is not None:
        try:
            _face_mesh.close()
        finally:
            _face_mesh = None

    if _face_detector is not None:
        try:
            _face_detector.close()
        finally:
            _face_detector = None


def _run_preview(camera_index: int = 0) -> None:
    """Standalone preview for terminal debugging only."""
    if not initialize(camera_index):
        raise RuntimeError(f"Failed to initialize camera index {camera_index}")

    print("Camera opened. Press ESC to exit.")

    try:
        while True:
            frame_bytes, height, width = capture_frame()
            if height == 0 or width == 0:
                break

            frame_rgb = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            landmarks = detect_landmarks(frame_bytes, height, width)

            cv2.putText(
                frame_bgr,
                f"Landmarks: {len(landmarks) // 3}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.imshow("MediaPipe DMS preview", frame_bgr)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_preview(camera_index=0)
