#!/usr/bin/env bash
# Run the classifier interpreter with its pip-installed CUDA libraries visible.
set -euo pipefail

ENV_DIR=${DISHWASHING_CLASSIFIER_ENV:-/root/.venvs/dishwashing-classifier}
PYTHON="$ENV_DIR/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Classifier environment not found: $ENV_DIR" >&2
    exit 1
fi

unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
shopt -s nullglob
NVIDIA_LIB_DIRS=("$ENV_DIR"/lib/python*/site-packages/nvidia/*/lib)
if ((${#NVIDIA_LIB_DIRS[@]})); then
    CUDA_LIBRARY_PATH=$(IFS=:; echo "${NVIDIA_LIB_DIRS[*]}")
    export LD_LIBRARY_PATH="$CUDA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PATH="$ENV_DIR/bin:$PATH"

exec "$PYTHON" "$@"
