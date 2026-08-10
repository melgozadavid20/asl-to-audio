# ASL Project

Camera reads an ASL sign -> classifier identifies it -> audio output speaks it.

Currently targeting a small vocabulary of essential words (hello, goodbye,
restroom, help, etc.) rather than full alphabet fingerspelling. Because most
of these signs involve movement, the classifier looks at a short clip of the
gesture (about a second, resampled to a fixed number of frames) rather than
a single still frame.

## Pipeline

1. **Collect data** — record labeled gesture clips for each sign you want to recognize.
2. **Train** — fit a classifier on the collected clips.
3. **Run** — live webcam recognition with spoken audio output.

## Setup

Requires Python 3.9-3.12 (mediapipe doesn't support 3.13+ yet) and macOS (audio output uses the built-in `say` command).

If you don't already have Python 3.11 and Homebrew set up:

```bash
xcode-select --install   # required for Homebrew; a GUI installer will pop up, let it finish
brew install python@3.11
```

Then create the venv and install dependencies:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Troubleshooting

**Camera shows black, fails to open, or grabs your iPhone instead of your Mac.**
macOS doesn't always assign your built-in webcam to index 0, especially with Continuity Camera (using an iPhone as a webcam) enabled. Every script here accepts a `--camera N` flag. Find the right index with:

```bash
python3 -c "
import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    ok, frame = cap.read()
    print(f'camera {i}: opened={cap.isOpened()} read_ok={ok} brightness={frame.mean() if ok else None}')
    cap.release()
"
```

Use whichever index reports a real (non-zero, non-`None`) brightness, e.g. `python src/collect_data.py --label HELLO --camera 1`. If it keeps grabbing your iPhone no matter the index, turn off Continuity Camera: iPhone Settings → General → AirPlay & Handoff → "Continuity Camera" (off), or Mac System Settings → General → AirDrop & Handoff → "Continuity Camera" (off). Also confirm Terminal has camera access under System Settings → Privacy & Security → Camera.

**No sound.** Test macOS's TTS directly with `say "test"` in Terminal. If that's silent, it's a system audio/volume issue, not this project.

## Usage

### 1. Collect training data

```bash
python src/collect_data.py --label HELLO
```

Get your hand ready, press `SPACE` to start recording one clip, perform the
sign, then either drop your hand out of frame or hold still until it
auto-stops (~2.5s max). Repeat per sample. `q` quits. Aim for 40+ samples per
sign, varying speed/angle/position a bit each time.

Repeat for every sign in your vocabulary, e.g.:

```bash
python src/collect_data.py --label HELLO
python src/collect_data.py --label GOODBYE
python src/collect_data.py --label RESTROOM
python src/collect_data.py --label HELP
```

Samples are saved as individual `.npy` files under `data/sequences/`.

### 2. Train the classifier

```bash
python src/train.py
```

Trains on everything in `data/sequences/`, prints validation accuracy, and
saves the model to `models/asl_classifier.pkl`.

### 3. Run live recognition + audio output

```bash
python src/predict_and_speak.py
```

Opens the webcam. When your hand appears, it starts recording; when your
hand drops out of frame (or after ~2.5s), it classifies the clip and speaks
the result out loud. It waits for your hand to be fully out of frame before
listening for the next sign, so one gesture doesn't get chopped into
multiple guesses. Press `q` to quit.

## Project layout

```
ASL Project/
  data/
    sequences/            # collected gesture clips, one .npy per sample (gitignored)
  models/                 # trained classifier (gitignored)
  src/
    capture.py             # webcam + MediaPipe hand landmark extraction (single frame)
    sequence_utils.py      # resampling + feature extraction for gesture clips
    collect_data.py        # labeled gesture data collection tool
    train.py                # trains + saves the classifier
    predict_and_speak.py    # real-time inference + TTS
  requirements.txt
```

## Notes / next steps

- This is a standalone camera-to-audio project -- no hardware/servo
  integration, just webcam in, spoken word out.
- Starter vocabulary suggestion: HELLO, GOODBYE, RESTROOM, HELP, YES, NO,
  PLEASE, THANK YOU. Keep it small at first, confirm accuracy, then expand.
- Real ASL fingerspelling (A-Z) is a different mode -- used for spelling
  names/proper nouns, not general conversation. If you want that later too,
  it fits the same sequence pipeline (letters just need shorter/simpler clips).
- Audio output uses macOS's built-in `say` command. Swap it for a cloud TTS
  voice (e.g. ElevenLabs, Google TTS) later if you want higher quality output
  or want this to run on non-Mac hardware eventually.
