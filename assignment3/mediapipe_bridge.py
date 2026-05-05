"""
Anna Roma - s345819
Assignment 3 - Driver Monitoring System

mediapipe_bridge.py - Matlab calls these functions to interface with MediaPipe FaceMesh in Python.
Handles webcam capture and MediaPipe Face Mesh landmark detection.
Returns raw RGB bytes and landmark coordinates [x y z] as Python primitives for easy use in Matlab.
"""
import cv2
import mediapipe as mp
import numpy as np 
import time
import statistics as st
import os

_cap = None
_face_mesh = None


def initialize(camera_index=0):
    """Open webcam and create FaceMesh. Call once from MATLAB before the main loop."""
    global _cap, _face_mesh
    _face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True, # create a model with iris landmarks MediaPipe FaceMesh
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _cap = cv2.VideoCapture(int(camera_index))
    _cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return bool(_cap.isOpened())


def capture_frame():
    """
    Grab one frame from the webcam.
    Returns (frame_bytes, height, width) as Python primitives.
    frame_bytes: bytes of RGB image in row-major order (H x W x 3).
    MATLAB reconstructs: permute(reshape(uint8(py.array.array('B',fb)), [3,w,h]), [3,2,1])
    """
    global _cap
    ret, frame = _cap.read()
    if not ret:
        return b'', 0, 0
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    return frame_rgb.tobytes(), int(h), int(w)


def detect_landmarks(frame_bytes, height, width):
    """
    Run MediaPipe FaceMesh on the given raw RGB bytes.
    Returns flat list [x0,y0,z0, x1,y1,z1, ...] for 478 landmarks (normalised 0-1),
    or empty list when no face is detected.
    """
    global _face_mesh
    h, w = int(height), int(width)
    arr = np.frombuffer(bytes(frame_bytes), dtype=np.uint8).reshape(h, w, 3)
    results = _face_mesh.process(arr)
    if results.multi_face_landmarks:
        coords = []
        for lm in results.multi_face_landmarks[0].landmark:
            coords.extend([float(lm.x), float(lm.y), float(lm.z)])
        return coords
    return []


def release():
    """Release webcam. Call once from MATLAB at shutdown."""
    global _cap
    if _cap is not None:
        _cap.release()
        _cap = None
