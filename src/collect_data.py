"""
Collect labeled gesture samples for training the ASL classifier.

Each sample is a short recording of a hand sign (a few frames of movement,
not just one still pose), saved as a .npy file under data/sequences/.

Usage:
    python src/collect_data.py --label HELLO
    python src/collect_data.py --label RESTROOM --samples 40

Controls:
    SPACE - start recording one gesture sample (get your hand ready first)
    q     - quit

Recording stops automatically when your hand leaves the frame, or after
a couple seconds -- whichever comes first. Perform the sign once per
recording (e.g. one wave for "hello"), then let your hand drop out of
frame or pause to end it.
"""

import argparse
import os
import time

import cv2
import numpy as np

from capture import HandLandmarker, open_camera
from sequence_utils import MAX_RECORD_SECONDS, MIN_FRAMES, SEQUENCE_LENGTH, resample_sequence

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sequences")

FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    parser = argparse.ArgumentParser(description="Collect labeled ASL gesture samples.")
    parser.add_argument("--label", required=True, help="Sign label for this session, e.g. HELLO, RESTROOM")
    parser.add_argument("--samples", type=int, default=40, help="Target number of samples to collect")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    landmarker = HandLandmarker()
    cap = open_camera(args.camera)

    collected = 0
    recording = False
    buffer = []
    record_start = 0.0

    print(f"Collecting samples for '{args.label}'. Get your hand ready, press SPACE to record, q to quit.")

    try:
        while collected < args.samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera.")
                break

            frame = cv2.flip(frame, 1)
            results, landmarks = landmarker.process(frame)
            display = landmarker.draw(frame.copy(), results)
            now = time.time()

            if recording:
                elapsed = now - record_start
                hand_gone = landmarks is None
                if not hand_gone:
                    buffer.append(landmarks)

                if hand_gone or elapsed >= MAX_RECORD_SECONDS:
                    if len(buffer) >= MIN_FRAMES:
                        seq = resample_sequence(buffer, SEQUENCE_LENGTH)
                        fname = f"{args.label}__{int(now * 1000)}.npy"
                        np.save(os.path.join(DATA_DIR, fname), seq)
                        collected += 1
                        print(f"Saved sample {collected}/{args.samples} ({len(buffer)} frames)")
                    else:
                        print("Too short / hand lost too fast -- discarded, try again.")
                    recording = False
                    buffer = []

                status = f"RECORDING  {elapsed:.1f}s"
                color = (0, 0, 255)
            else:
                status = f"label={args.label}  collected={collected}/{args.samples}  SPACE=record  q=quit"
                color = (0, 255, 0)

            cv2.putText(display, status, (10, 30), FONT, 0.7, color, 2)
            hand_status = "hand detected" if landmarks is not None else "no hand"
            cv2.putText(display, hand_status, (10, 60), FONT, 0.7,
                        (0, 255, 0) if landmarks is not None else (0, 0, 255), 2)

            cv2.imshow("Collect ASL Data", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" ") and not recording:
                if landmarks is None:
                    print("No hand detected -- get your hand in frame first.")
                else:
                    recording = True
                    buffer = [landmarks]
                    record_start = now

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    print(f"Done. {collected} samples for '{args.label}' saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
