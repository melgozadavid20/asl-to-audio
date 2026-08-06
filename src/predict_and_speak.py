"""
Real-time ASL sign recognition with spoken audio output.

Automatically starts "recording" a gesture when your hand appears in frame,
stops when your hand drops out of frame (or after a couple seconds), then
classifies the whole clip and speaks the result. Waits for your hand to be
fully out of frame before it will start listening for the next sign, so one
gesture doesn't get chopped into multiple guesses.

Usage:
    python src/predict_and_speak.py
"""

import os
import subprocess
import time

import cv2
import joblib

from capture import HandLandmarker, open_camera
from sequence_utils import MAX_RECORD_SECONDS, MIN_FRAMES, resample_sequence, sequence_to_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "asl_classifier.pkl")

# How long the hand must be fully out of frame after a prediction before
# we'll start listening for the next gesture. Stops one sign from getting
# split into several predictions if the hand lingers near the edge.
HAND_ABSENT_RESET_SECONDS = 0.3

FONT = cv2.FONT_HERSHEY_SIMPLEX

STATE_IDLE = "idle"
STATE_RECORDING = "recording"
STATE_COOLDOWN = "cooldown"


class Speaker:
    """Speaks text using macOS's built-in 'say' command.

    We tried pyttsx3 first, but on macOS its speech engine (NSSpeechSynthesizer)
    needs to run on the main thread with an active run loop -- calling it from a
    background thread (needed so speaking doesn't freeze the video loop) produces
    no audio and no error. Shelling out to 'say' sidesteps that entirely: it's a
    separate process, so there's no thread/run-loop issue, and it's already on
    every Mac.
    """

    def say(self, text):
        subprocess.Popen(["say", str(text)])


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run collect_data.py then train.py first."
        )

    model_data = joblib.load(MODEL_PATH)
    clf = model_data["model"]
    sequence_length = model_data["sequence_length"]

    landmarker = HandLandmarker()
    cap = open_camera(0)
    speaker = Speaker()

    state = STATE_IDLE
    buffer = []
    record_start = 0.0
    absent_since = None
    last_prediction = None

    print("Running. Show a sign, then drop your hand to hear it spoken. Press q to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera.")
                break

            frame = cv2.flip(frame, 1)
            results, landmarks = landmarker.process(frame)
            display = landmarker.draw(frame.copy(), results)
            now = time.time()

            if state == STATE_IDLE:
                if landmarks is not None:
                    state = STATE_RECORDING
                    buffer = [landmarks]
                    record_start = now

            elif state == STATE_RECORDING:
                elapsed = now - record_start
                hand_gone = landmarks is None
                if not hand_gone:
                    buffer.append(landmarks)

                if hand_gone or elapsed >= MAX_RECORD_SECONDS:
                    if len(buffer) >= MIN_FRAMES:
                        seq = resample_sequence(buffer, sequence_length)
                        features = sequence_to_features(seq)
                        prediction = clf.predict([features])[0]
                        speaker.say(prediction)
                        last_prediction = prediction
                    state = STATE_COOLDOWN
                    absent_since = now if hand_gone else None
                    buffer = []

            elif state == STATE_COOLDOWN:
                if landmarks is None:
                    if absent_since is None:
                        absent_since = now
                    elif now - absent_since >= HAND_ABSENT_RESET_SECONDS:
                        state = STATE_IDLE
                        absent_since = None
                else:
                    absent_since = None  # hand still in frame, keep waiting for it to drop

            # --- overlay ---
            state_label = {
                STATE_IDLE: "waiting for hand...",
                STATE_RECORDING: "RECORDING sign...",
                STATE_COOLDOWN: "drop your hand to reset",
            }[state]
            color = (0, 0, 255) if state == STATE_RECORDING else (0, 255, 0)
            cv2.putText(display, state_label, (10, 30), FONT, 0.8, color, 2)
            if last_prediction:
                cv2.putText(display, f"Last sign: {last_prediction}", (10, 60), FONT, 0.8, (0, 200, 255), 2)
            cv2.putText(display, "q=quit", (10, 90), FONT, 0.6, (255, 255, 255), 1)

            cv2.imshow("ASL to Audio", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
