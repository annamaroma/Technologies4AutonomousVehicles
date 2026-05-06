"""
Anna Roma - s345819
Assignment 3 - Driver Monitoring System

mediapipe_bridge.py - Matlab calls these functions to interface with MediaPipe FaceMesh in Python.
Handles webcam capture and MediaPipe Face Mesh landmark detection.
Returns raw RGB bytes and landmark coordinates [x y z] as Python primitives for easy use in Matlab.
"""
# mediapipe_bridge.py
import cv2
import mediapipe as mp
import numpy as np
import time

# Same landmark groups used in the professor sample
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249,
            263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155,
             133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [473, 474, 475, 476, 477]
RIGHT_IRIS = [468, 469, 470, 471, 472]
NOSE_TIP = [45, 4, 275]

mp_face_mesh = mp.solutions.face_mesh


def create_face_mesh():
    """Create MediaPipe FaceMesh, same style as the professor sample."""
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # enables iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def process_frame(face_mesh, frame_bgr):
    """
    Input: OpenCV BGR frame.
    Output:
        frame_out: BGR frame with selected landmarks drawn
        landmarks: numpy array [N x 3] with normalized x,y,z, or None
    """
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    image_rgb.flags.writeable = False
    results = face_mesh.process(image_rgb)
    image_rgb.flags.writeable = True

    frame_out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    img_h, img_w = frame_out.shape[:2]

    landmarks = None

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
            dtype=np.float32
        )

        for idx, lm in enumerate(face_landmarks.landmark):
            x = int(lm.x * img_w)
            y = int(lm.y * img_h)

            if idx in LEFT_EYE or idx in RIGHT_EYE:
                cv2.circle(frame_out, (x, y), 2, (0, 0, 255), -1)

            if idx in LEFT_IRIS or idx in RIGHT_IRIS:
                cv2.circle(frame_out, (x, y), 2, (0, 255, 0), -1)

            if idx in NOSE_TIP:
                cv2.circle(frame_out, (x, y), 2, (255, 0, 0), -1)

    return frame_out, landmarks


def run_preview(camera_index=0):
    """
    Standalone test: opens webcam and shows MediaPipe landmarks.
    Use this from terminal, not from MATLAB pyrun.
    """
    face_mesh = create_face_mesh()

    # On your Linux machine index 0 works; CAP_V4L2 makes backend explicit
    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {camera_index}")

    print(f"Camera {camera_index} opened. Press ESC to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        start = time.time()

        frame_out, landmarks = process_frame(face_mesh, frame)

        fps = 1.0 / max(time.time() - start, 1e-6)
        n_landmarks = 0 if landmarks is None else landmarks.shape[0]

        cv2.putText(
            frame_out,
            f"Landmarks: {n_landmarks} | FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("MediaPipe DMS preview", frame_out)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    run_preview(camera_index=0)