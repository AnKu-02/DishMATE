#!/usr/bin/env bash
# Enable TensorFlow GPU support in the separate WSL classifier environment.
set -euo pipefail

ENV_DIR=${DISHWASHING_CLASSIFIER_ENV:-/root/.venvs/dishwashing-classifier}
PYTHON="$ENV_DIR/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Classifier environment not found: $ENV_DIR" >&2
    echo "Create it first with: bash setup/setup_classifier_env.sh" >&2
    exit 1
fi

unset PYTHONPATH PYTHONHOME VIRTUAL_ENV

# TensorFlow's Linux/WSL extra installs the matching CUDA and cuDNN wheels.
"$PYTHON" -m pip install --upgrade 'tensorflow[and-cuda]==2.21.0' 'keras==3.12.1'
"$PYTHON" -m pip check

# TensorFlow documents these links as the fallback when the NVIDIA wheels are
# installed but their shared libraries or ptxas are not discovered in a venv.
TF_DIR=$("$PYTHON" -c 'import os, tensorflow; print(os.path.dirname(tensorflow.__file__))')
shopt -s nullglob
NVIDIA_LIBS=("$TF_DIR"/../nvidia/*/lib/*.so*)
if ((${#NVIDIA_LIBS[@]})); then
    ln -sf "${NVIDIA_LIBS[@]}" "$TF_DIR/"
fi

NVIDIA_LIB_DIRS=("$TF_DIR"/../nvidia/*/lib)
CUDA_LIBRARY_PATH=$(IFS=:; echo "${NVIDIA_LIB_DIRS[*]}")
export LD_LIBRARY_PATH="$CUDA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

NVCC_MODULE=$("$PYTHON" -c 'import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__file__)')
PTXAS=$(find "$(dirname "$(dirname "$NVCC_MODULE")")" -path '*/bin/ptxas' -type f -print -quit)
if [[ -n "$PTXAS" ]]; then
    ln -sf "$PTXAS" "$ENV_DIR/bin/ptxas"
fi

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$PROJECT_DIR/setup/run_classifier_python.sh" "$PROJECT_DIR/tests/check_tensorflow_cuda.py"
