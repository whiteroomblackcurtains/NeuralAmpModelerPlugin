import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from NeuralPitchModeler.pitch_model import prepare_training_data


class TrainingDataTests(unittest.TestCase):
    def test_builds_multilabel_windows_for_overlapping_notes(self):
        sample_rate = 22050
        duration = 2.0
        time = np.arange(int(sample_rate * duration)) / sample_rate
        audio = (0.2 * np.sin(2 * np.pi * 110 * time) * (2**30)).astype(np.int32)

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "guitar.wav"
            notes_path = Path(temp_dir) / "notes.json"
            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(4)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio.tobytes())
            notes_path.write_text(
                json.dumps(
                    {
                        "notes": [
                            {
                                "start_seconds": 0.25,
                                "end_seconds": 1.5,
                                "midi_pitch": 45,
                            },
                            {
                                "start_seconds": 0.75,
                                "end_seconds": 1.25,
                                "midi_pitch": 52,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            data = prepare_training_data(
                audio_path,
                notes_path,
                window_seconds=1.0,
                hop_seconds=0.1,
                midi_min=40,
                midi_max=60,
            )

        self.assertEqual(data.features.ndim, 4)
        self.assertEqual(data.labels.shape, (len(data.features), 21))
        self.assertEqual(data.active_pitch_count, 2)
        overlap_rows = np.logical_and(
            data.labels[:, 45 - 40] == 1.0,
            data.labels[:, 52 - 40] == 1.0,
        )
        self.assertTrue(np.any(overlap_rows))
        self.assertAlmostEqual(data.features.mean(), 0.0, places=4)
        self.assertAlmostEqual(data.features.std(), 1.0, places=4)

    def test_rejects_note_outside_audio_duration(self):
        sample_rate = 22050
        audio = np.zeros(sample_rate, dtype=np.int16)
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "guitar.wav"
            notes_path = Path(temp_dir) / "notes.json"
            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio.tobytes())
            notes_path.write_text(
                json.dumps(
                    {
                        "notes": [
                            {"start_seconds": 0.5, "end_seconds": 1.5, "midi_pitch": 45}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "beyond the audio duration"):
                prepare_training_data(
                    audio_path,
                    notes_path,
                    window_seconds=1.0,
                    hop_seconds=0.1,
                    midi_min=40,
                    midi_max=60,
                )


if __name__ == "__main__":
    unittest.main()
