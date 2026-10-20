# Neural Pitch Modeler

This is an experimental offline trainer for supervised, polyphonic guitar pitch recognition. It learns a multi-label MIDI-pitch estimate from a WAV recording and a JSON file containing the notes played at each time. The model is fitted to the supplied recording; it does not shift audio or yet provide a plug-in inference path.

## Install

From the repository root, install the Python dependencies:

```bash
python3 -m pip install -r NeuralPitchModeler/requirements.txt
```

The pinned PyTorch package is CPU-only. To train with CUDA, install the matching CUDA-enabled PyTorch package for your system using the [official PyTorch instructions](https://pytorch.org/get-started/locally/) instead.

## Annotation format

The JSON file must contain a `notes` array. Each note event uses seconds from the start of the WAV and an integer MIDI pitch. Events may overlap to represent chords:

```json
{
  "notes": [
    {"start_seconds": 0.0, "end_seconds": 0.8, "midi_pitch": 40},
    {"start_seconds": 0.2, "end_seconds": 0.9, "midi_pitch": 47}
  ]
}
```

Intervals include their start time and exclude their end time. Annotations must fall within the audio duration and configured MIDI pitch range. Mono or stereo PCM/float WAV files are accepted; stereo channels are averaged to mono.

## Train a model

```bash
python3 NeuralPitchModeler/pitch_model.py \
  --audio guitar.wav \
  --notes notes.json \
  --output guitar_pitch_model.pt \
  --window-seconds 1.0
```

`--window-seconds` sets the audio context used for each pitch estimate. `--hop-seconds` sets the spacing between estimates (default 0.05 seconds). The output is a PyTorch checkpoint containing the model weights and audio/pitch configuration. MIDI output defaults to 36–96; use `--midi-min` and `--midi-max` to change the range. Training duration, batch size, learning rate, random seed, and device can also be configured; run the script with `--help` for the full command-line options.

The training target for each example is the set of MIDI pitches active at the center time of its analysis window. Output at inference is therefore a set of per-time pitch probabilities; a later inference application can threshold and group adjacent estimates into note events.

## Tests

```bash
python3 -m unittest discover -s NeuralPitchModeler/tests -v
```

This prototype reports training loss only. Overlapping windows from a single recording are highly correlated, so a low training loss does not establish that the model generalizes to new performances, guitars, tunings, or recording conditions. Test it on separately recorded and annotated material before relying on its predictions.
