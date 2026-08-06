"""
Shared webcam + MediaPipe Hands utilities used by both data collection
and real-time inference.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


class HandLandmarker:
    """Wraps MediaPipe Hands for single-hand landmark extraction."""

    def __init__(self, max_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        """Returns (results, landmarks_array_or_None).

        landmarks_array is a flattened, normalized (x, y, z) * 21 numpy array
        relative to the hand's own bounding box, so it's roughly invariant to
        where the hand is in the frame and how far it is from the camera.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return results, None

        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = normalize_landmarks(hand_landmarks)
        return results, landmarks

    def draw(self, frame_bgr, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
        return frame_bgr

    def close(self):
        self.hands.close()


def normalize_landmarks(hand_landmarks):
    """Flatten MediaPipe's 21 (x, y, z) landmarks into a normalized feature vector.

    Normalization: translate so the wrist (landmark 0) is the origin, then scale
    by the max distance from the wrist to any other landmark. This makes the
    features roughly invariant to hand position in frame and distance from camera,
    which matters a lot for a small classifier to generalize well.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    coords = coords - wrist
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 1e-6:
        coords = coords / scale
    return coords.flatten()  # shape (63,)


def open_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    return cap
