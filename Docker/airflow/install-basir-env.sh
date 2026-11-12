#!/usr/bin/env bash
# Build Basir's environment at $UV_PROJECT_ENVIRONMENT from uv.lock, with the CPU torch wheel.
#
# Same approach as .github/scripts/sync-cpu-env.sh: every locked package except torch and its
# CUDA libraries (~2.9 GB), then the same torch version from the PyTorch CPU index. Pipeline
# steps that need the GPU will not run here (see docs/07-pipelines.md).
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
uv sync --frozen --no-dev --no-install-project "${skip[@]}"
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps "torch==${torch_version}" \
  --index-url https://download.pytorch.org/whl/cpu
"$UV_PROJECT_ENVIRONMENT/bin/python" -c "import cv2, mlflow, open3d, torch; print('torch', torch.__version__)"
