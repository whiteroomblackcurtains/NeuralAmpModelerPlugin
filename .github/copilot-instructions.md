# Project Guidelines

## Build and Test

- The plug-in is built with the iPlug2 platform projects and distribution scripts, not a root-level CMake build. The CI build in `.github/workflows/build-native.yml` is the reference for supported native builds.
- Initialize submodules recursively and download the iPlug2 SDKs and prebuilt libraries before building, as shown in the workflow. On macOS, run `cd NeuralAmpModeler/scripts && ./makedist-mac.sh full zip`; on Windows, run `cd NeuralAmpModeler\scripts` followed by `.\makedist-win.bat full zip`. These builds require their platform toolchains.
- The NeuralAmpModelerCore tools can be built and tested with:

  ```bash
  cmake -S NeuralAmpModelerCore -B NeuralAmpModelerCore/build -DCMAKE_BUILD_TYPE=Debug
  cmake --build NeuralAmpModelerCore/build -j4
  ./NeuralAmpModelerCore/build/tools/run_tests
  ```

  `run_tests` is a single executable whose test calls are listed in `NeuralAmpModelerCore/tools/run_tests.cpp`; it has no test-name filter. To run one case, temporarily leave only that case's function call in `main()` (for example, `test_dsp::test_construct()`), then rebuild and run the same executable.
- AudioDSPTools provides a CTest case for WAV path handling:

  ```bash
  cmake -S AudioDSPTools -B AudioDSPTools/build -DCMAKE_BUILD_TYPE=Debug
  cmake --build AudioDSPTools/build -j4
  ctest --test-dir AudioDSPTools/build -R wav_unicode_path --output-on-failure
  ```

- The experimental polyphonic pitch trainer is a separate Python tool. Install dependencies and train a model with:

  ```bash
  python3 -m pip install -r NeuralPitchModeler/requirements.txt
  python3 NeuralPitchModeler/pitch_model.py --audio guitar.wav --notes notes.json --output guitar_pitch_model.pt
  ```

  Run its tests with `python3 -m unittest discover -s NeuralPitchModeler/tests -v`; run one test with `python3 -m unittest NeuralPitchModeler.tests.test_pitch_model.TrainingDataTests.test_rejects_note_outside_audio_duration`.
- Format tracked plug-in C++ files with `bash format.bash` (requires `clang-format`). The configured pre-commit checks can be run with `pre-commit run --all-files`.
- The plug-in has no dedicated unit-test target. `CONTRIBUTING.md` lists the manual checks expected for plug-in changes, including model/IR loading, audio behavior, VST3 validation, and AU builds.

## Architecture

- `NeuralAmpModeler/` contains the iPlug2 plug-in implementation and per-platform app/VST3/AU project files. `NeuralAmpModeler.cpp` owns plug-in setup, UI layout, parameter handling, and the audio callback; `NeuralAmpModelerControls.h` contains custom UI controls.
- The root `iPlug2`, `NeuralAmpModelerCore`, `AudioDSPTools`, and `eigen` directories are submodules. The plug-in wraps NAM Core models in `ResamplingNAM`; AudioDSPTools supplies supporting processing such as impulse responses, resampling, noise gating, and WAV handling.
- The audio callback processes internally as mono: input conversion/gain, NAM inference, noise gate, tone stack, optional IR, DC-blocking high-pass filter, then output conversion/gain and meters. Model and IR loading is staged separately and applied at an audio-block boundary; preserve that handoff rather than constructing or destroying DSP modules in the audio path.
- Plug-in state stores parameter values and model/IR paths, not model data. `Unserialization.cpp` converts older versioned state layouts to the current configuration.
- `NeuralPitchModeler/` is a separate experimental PyTorch trainer for polyphonic pitch recognition. It learns framewise multi-label MIDI pitches from WAV audio and time-aligned JSON note events; it is not part of NAM Core inference or the plug-in audio callback.

## Conventions

- `EParams` in `NeuralAmpModeler/NeuralAmpModeler.h` defines parameter indices; the first six indices also determine knob layout. Treat parameter order and serialized names as compatibility-sensitive. When changing parameters, update the versioned migration logic in `Unserialization.cpp` and check that older saved states still load.
- When adding or changing graphical resources, define their names in `NeuralAmpModeler/config.h` and register them in the required locations in `NeuralAmpModeler/resources/main.rc`.
- C++ formatting follows the repository's LLVM-based `.clang-format` via `bash format.bash`. See `CONTRIBUTING.md` for the project contribution and manual validation checklist.
