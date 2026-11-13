#!/usr/bin/env bash
# Build the project venv from uv.lock, but with the CPU-only torch wheel instead of the CUDA one.
#
# uv.lock takes torch from PyPI, whose Linux wheel pulls ~2.9 GB of CUDA libraries (nvidia-*, cuda-*,
# triton). CI runners have no GPU and the tests never touch CUDA, so we install every locked package
# except those, then the same torch version from the PyTorch CPU index (~210 MB). The local CUDA
# source in pyproject.toml / uv.lock is unchanged. Callers must set UV_NO_SYNC=1 afterwards so
# `uv run` does not "repair" the venv back to the CUDA wheel.
set -euo pipefail

lock=uv.lock
torch_version=$(grep -A1 '^name = "torch"$' "$lock" | sed -n 's/^version = "\(.*\)"$/\1/p')
if [[ -z "$torch_version" ]]; then
  echo "torch not found in $lock" >&2
  exit 1
fi

skip=()
while read -r pkg; do
  skip+=(--no-install-package "$pkg")
done < <(grep -oE '^name = "(torch|triton|nvidia-[^"]+|cuda-[^"]+)"$' "$lock" | cut -d'"' -f2)

echo "Skipping ${#skip[@]} CUDA-side packages; installing torch==${torch_version} from the CPU index"
uv sync --frozen "${skip[@]}"
uv pip install --no-deps "torch==${torch_version}" --index-url https://download.pytorch.org/whl/cpu
uv run --no-sync python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
