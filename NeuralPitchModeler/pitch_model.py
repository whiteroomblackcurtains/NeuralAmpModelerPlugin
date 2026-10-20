#!/usr/bin/env python3
"""Train a supervised, multi-pitch guitar model from WAV and note annotations."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy.io import wavfile
from scipy.interpolate import interp1d
from scipy.signal import resample_poly, stft
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TARGET_SAMPLE_RATE = 22050
FFT_SIZE = 8192
HOP_LENGTH = 220
LOG_FREQ_BINS = 1024
MIN_FEATURE_FREQUENCY = 45.0
MAX_FEATURE_FREQUENCY = 10000.0
ARCHITECTURE_VERSION = 1


class PitchModel(nn.Module):
    """Predict active MIDI pitches while retaining frequency and time positions."""

    def __init__(self, num_pitches: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=(1, 2), padding=2),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((64, 7)),
            nn.Flatten(),
            nn.Linear(32 * 64 * 7, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(256, num_pitches)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(features))


@dataclass
class TrainingData:
    features: np.ndarray
    labels: np.ndarray
    feature_mean: float
    feature_std: float
    pitches: Sequence[int]
    duration_seconds: float
    active_pitch_count: int
    window_seconds: float
    hop_seconds: float


def _read_audio(path: Path) -> np.ndarray:
    sample_rate, audio = wavfile.read(str(path))
    if audio.ndim == 2:
        audio = audio.astype(np.float64).mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError("Audio must be mono or stereo PCM/float WAV.")

    if np.issubdtype(audio.dtype, np.integer):
        limits = np.iinfo(audio.dtype)
        audio = audio.astype(np.float64) / max(abs(limits.min), limits.max)
    else:
        audio = audio.astype(np.float64)

    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or contains non-finite samples.")
    if sample_rate <= 0:
        raise ValueError("Audio sample rate must be positive.")

    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        )

    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32)


def _read_note_events(
    path: Path, duration: float, midi_min: int, midi_max: int
) -> List[Dict[str, float]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read note annotations: {error}") from error

    if not isinstance(data, dict) or not isinstance(data.get("notes"), list):
        raise ValueError('Annotation JSON must be an object with a "notes" array.')
    events: List[Dict[str, float]] = []
    for index, note in enumerate(data["notes"]):
        if not isinstance(note, dict):
            raise ValueError(f"Note event {index} must be a JSON object.")
        try:
            start = float(note["start_seconds"])
            end = float(note["end_seconds"])
            pitch_value = note["midi_pitch"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Note event {index} needs start_seconds, end_seconds, and midi_pitch."
            ) from error

        if isinstance(pitch_value, bool) or not isinstance(pitch_value, int):
            raise ValueError(f"Note event {index} midi_pitch must be an integer.")
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            raise ValueError(
                f"Note event {index} must have finite times with 0 <= start < end."
            )
        if end > duration:
            raise ValueError(
                f"Note event {index} ends at {end:.3f}s, beyond the audio duration of {duration:.3f}s."
            )
        if pitch_value < midi_min or pitch_value > midi_max:
            raise ValueError(
                f"Note event {index} MIDI pitch {pitch_value} is outside "
                f"the configured range {midi_min}..{midi_max}."
            )
        events.append({"start": start, "end": end, "pitch": pitch_value})

    if not events:
        raise ValueError("At least one annotated note event is required.")
    return events


def prepare_training_data(
    audio_path: Path,
    notes_path: Path,
    window_seconds: float,
    hop_seconds: float,
    midi_min: int,
    midi_max: int,
) -> TrainingData:
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("--window-seconds must be a positive number.")
    if not math.isfinite(hop_seconds) or hop_seconds <= 0:
        raise ValueError("--hop-seconds must be a positive number.")
    if midi_min < 0 or midi_max > 127 or midi_min > midi_max:
        raise ValueError(
            "MIDI range must satisfy 0 <= --midi-min <= --midi-max <= 127."
        )

    audio = _read_audio(audio_path)
    duration = len(audio) / TARGET_SAMPLE_RATE
    events = _read_note_events(notes_path, duration, midi_min, midi_max)
    if len(audio) < FFT_SIZE:
        audio = np.pad(audio, (0, FFT_SIZE - len(audio)))
    frequencies, _, spectrum = stft(
        audio,
        fs=TARGET_SAMPLE_RATE,
        nperseg=FFT_SIZE,
        noverlap=FFT_SIZE - HOP_LENGTH,
        nfft=FFT_SIZE,
        boundary="zeros",
        padded=True,
    )
    magnitude = np.abs(spectrum)
    log_frequencies = np.geomspace(
        MIN_FEATURE_FREQUENCY,
        MAX_FEATURE_FREQUENCY,
        LOG_FREQ_BINS,
    )
    spectrogram = interp1d(
        frequencies,
        magnitude,
        axis=0,
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )(log_frequencies)
    spectrogram = np.log1p(spectrogram).astype(np.float32)

    frame_step_seconds = HOP_LENGTH / TARGET_SAMPLE_RATE
    frame_count = max(1, int(round(window_seconds / frame_step_seconds)))
    if frame_count % 2 == 0:
        frame_count += 1
    half_window = frame_count // 2

    centers = np.arange(0.0, duration, hop_seconds, dtype=np.float64)
    centers = np.unique(
        np.concatenate(
            (
                centers,
                np.asarray([(event["start"] + event["end"]) / 2.0 for event in events]),
            )
        )
    )
    features = np.empty(
        (len(centers), 1, spectrogram.shape[0], frame_count),
        dtype=np.float32,
    )
    labels = np.zeros((len(centers), midi_max - midi_min + 1), dtype=np.float32)
    for row, center in enumerate(centers):
        center_frame = int(round(center / frame_step_seconds))
        frame_indices = np.arange(
            center_frame - half_window, center_frame + half_window + 1
        )
        valid = (frame_indices >= 0) & (frame_indices < spectrogram.shape[1])
        window = np.zeros((spectrogram.shape[0], frame_count), dtype=np.float32)
        window[:, valid] = spectrogram[:, frame_indices[valid]]
        features[row, 0] = window
        for event in events:
            if event["start"] <= center < event["end"]:
                labels[row, int(event["pitch"]) - midi_min] = 1.0

    feature_mean = float(features.mean())
    feature_std = max(float(features.std()), 1e-6)
    features = (features - feature_mean) / feature_std
    pitches = tuple(range(midi_min, midi_max + 1))
    return TrainingData(
        features=features,
        labels=labels,
        feature_mean=feature_mean,
        feature_std=feature_std,
        pitches=pitches,
        duration_seconds=duration,
        active_pitch_count=int(np.count_nonzero(labels.sum(axis=0))),
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    )


def train(
    data: TrainingData,
    output_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str,
) -> float:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("Epochs, batch size, and learning rate must be positive.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but it is not available.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    x = torch.from_numpy(data.features)
    y = torch.from_numpy(data.labels)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    model = PitchModel(num_pitches=len(data.pitches)).to(device)
    positives = y.sum(dim=0)
    negatives = len(y) - positives
    positive_weights = (
        (negatives / positives.clamp(min=1)).clamp(min=1.0, max=20.0).to(device)
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        total_loss = 0.0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_features)
        final_loss = total_loss / len(data.features)

    checkpoint = {
        "architecture_version": ARCHITECTURE_VERSION,
        "model_state_dict": model.cpu().state_dict(),
        "config": {
            "sample_rate": TARGET_SAMPLE_RATE,
            "fft_size": FFT_SIZE,
            "hop_length": HOP_LENGTH,
            "log_frequency_bins": LOG_FREQ_BINS,
            "min_feature_frequency": MIN_FEATURE_FREQUENCY,
            "max_feature_frequency": MAX_FEATURE_FREQUENCY,
            "window_seconds": data.window_seconds,
            "hop_seconds": data.hop_seconds,
            "window_frames": int(data.features.shape[-1]),
            "midi_min": int(data.pitches[0]),
            "midi_max": int(data.pitches[-1]),
            "feature_mean": data.feature_mean,
            "feature_std": data.feature_std,
            "model_type": "multi_label_spectrogram_cnn",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return final_loss


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a polyphonic MIDI-pitch model from annotated guitar audio."
    )
    parser.add_argument(
        "--audio", required=True, type=Path, help="Input PCM/float WAV file."
    )
    parser.add_argument(
        "--notes", required=True, type=Path, help="JSON note-event annotations."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output PyTorch checkpoint."
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=1.0,
        help="Audio context around each prediction time (default: 1 second).",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=0.05,
        help="Spacing between training prediction times (default: 0.05 seconds).",
    )
    parser.add_argument(
        "--midi-min", type=int, default=36, help="Lowest output MIDI pitch."
    )
    parser.add_argument(
        "--midi-max", type=int, default=96, help="Highest output MIDI pitch."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs (default: 50)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="AdamW learning rate."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Training device (default: cpu).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        data = prepare_training_data(
            args.audio,
            args.notes,
            args.window_seconds,
            args.hop_seconds,
            args.midi_min,
            args.midi_max,
        )
        loss = train(
            data,
            args.output,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.seed,
            args.device,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Audio duration: {data.duration_seconds:.2f} seconds")
    print(f"Training windows: {len(data.features)}")
    print(f"Annotated pitches present: {data.active_pitch_count}")
    print(f"Final training loss: {loss:.4f}")
    print(f"Saved model checkpoint: {args.output}")
    print(
        "This checkpoint is trained on this recording; it has not been validated on unseen audio."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
