"""
Shared helpers for treating a hand sign as a short sequence of frames
instead of a single still frame, so motion-based signs like "hello" or
"help" can be recognized -- not just static handshapes like a letter.
"""

import numpy as np

# Every recorded/live gesture gets resampled to this many frames before
# being turned into a feature vector, so training samples and live samples
# are always the same length regardless of how long the gesture took or
# what frame rate the camera happens to be running at.
SEQUENCE_LENGTH = 30

# Hard cap on how long a single gesture recording is allowed to run.
MAX_RECORD_SECONDS = 2.5

# Minimum number of raw frames required to keep a sample. Filters out
# accidental blips (hand flashing through frame for a split second).
MIN_FRAMES = 5


def resample_sequence(frames, target_len=SEQUENCE_LENGTH):
    """Resample a list/array of (63,) landmark frames to exactly target_len
    frames using linear interpolation, so gestures of different lengths and
    frame rates end up as the same-sized feature vector.
    """
    frames = np.asarray(frames, dtype=np.float64)
    n = len(frames)
    if n == target_len:
        return frames
    if n == 1:
        return np.repeat(frames, target_len, axis=0)

    old_idx = np.linspace(0, 1, num=n)
    new_idx = np.linspace(0, 1, num=target_len)
    resampled = np.empty((target_len, frames.shape[1]))
    for d in range(frames.shape[1]):
        resampled[:, d] = np.interp(new_idx, old_idx, frames[:, d])
    return resampled


def sequence_to_features(seq):
    """Turn a (SEQUENCE_LENGTH, 63) array of landmark positions into a flat
    feature vector containing both the raw positions and the frame-to-frame
    velocity, so the classifier can see the motion shape of the gesture and
    not just a blurry average of where the hand sat.
    """
    seq = np.asarray(seq, dtype=np.float64)
    velocity = np.diff(seq, axis=0)
    return np.concatenate([seq.flatten(), velocity.flatten()])
