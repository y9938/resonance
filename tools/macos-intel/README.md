# Intel macOS CPU environment

The installer creates `.deps/macos-intel`; `./r` and the desktop launcher
select this environment automatically on x86_64 macOS. Apple Silicon and
other platforms continue using the project's normal `.venv` workflow.

The native stack comes from conda-forge: Python 3.12.15, PyTorch 2.13.0 CPU,
torchaudio 2.11.0, TorchCodec 0.16.0, PyAV 18.1.0, FFmpeg 9.0.2, and NumPy
2.2.6. These torchaudio and TorchCodec builds link against PyTorch 2.13.0.
The project normally pins PyTorch 2.14.0, whose PyPI macOS wheels are ARM64.

PyTorch 2.2.2 from PyPI was rejected after reproducing `RuntimeError: Numpy
is not available` with NumPy 2. GigaAM requires NumPy 2. Numba 0.61.2 and
llvmlite 0.44.0 avoid building the newer LLVM runtime from source for Whisper.
PyAV also comes from conda-forge so that it shares FFmpeg with TorchCodec;
the PyPI PyAV wheel loaded a second copy of FFmpeg and produced duplicate
Objective-C class warnings.

## Homebrew on Intel

The [current Homebrew installer](https://github.com/Homebrew/install/blob/main/install.sh)
rejects Intel macOS; this change was
[merged on 2026-09-11](https://github.com/Homebrew/install/commit/fde1410a61157a71c78d2dc0c3a57a3a848b9756).
If Homebrew is missing, the historical
[installer revision `df3c25561`](https://github.com/Homebrew/install/commit/df3c2556141c012eaba0453ede7a1bd4f05b824c)
still supports installation into `/usr/local`:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/Homebrew/install/df3c25561/install.sh \
  -o /tmp/homebrew-install-intel.sh
/bin/bash /tmp/homebrew-install-intel.sh
eval "$(/usr/local/bin/brew shellenv)"
```

For an existing installation, only the `shellenv` step is needed when `brew`
is absent from PATH. This compatibility workaround downloads current Homebrew.
Intel macOS remains [Tier 3](https://docs.brew.sh/Support-Tiers#tier-3), and
updated formulae may need to compile from source because new Intel bottles
are no longer built.

## Environment

Use a native x86_64 shell and `DEVICE=cpu`.

To bootstrap without Homebrew, install `micromamba` and `uv` on PATH and run
`./r dev-deps`. The native lock also installs Node.js 22.23.2, uv 0.12.23,
micromamba 2.9.0, resvg 0.47.0 and ImageMagick 7.1.2-31 into the prefix.
The explicit lock is for Intel macOS only. Keep the native conda packages together.

The equivalent manual environment installation is:

```bash
micromamba create -y --prefix "$PWD/.deps/macos-intel" \
  --file tools/macos-intel/conda-osx-64.lock

uv pip install --python .deps/macos-intel/bin/python \
  -r pyproject.toml --group dev \
  --override tools/macos-intel/overrides.txt \
  --constraint tools/macos-intel/python-versions.txt

.deps/macos-intel/bin/python -m pip check
```

`environment.yml` describes the selected native versions;
`conda-osx-64.lock` records their exact package builds and dependency closure.
`python-versions.txt` records all installed Python versions and is used as
constraints after the native environment is created. It is not a standalone
pip requirements file: several native packages have no Intel macOS PyPI wheel.
`overrides.txt` replaces the project's PyTorch pin and fixes the tested
GigaAM source revision. `constraints.txt` contains the smaller compatibility
constraints used during initial selection.

## Compatibility

CPU inference was checked on macOS Sonoma 14.8.9 (x86_64) with GigaAM-v3,
Whisper Turbo, Silero and Kokoro. Full Granite inference and interactive
audio capture have not been validated with this profile.
