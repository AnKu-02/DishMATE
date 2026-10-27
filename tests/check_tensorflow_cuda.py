"""Verify the CUDA libraries and TensorFlow GPU execution in WSL."""

from __future__ import annotations

import ctypes


CUDA_LIBRARIES = (
    "libcuda.so.1",
    "libcudart.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcufft.so.11",
    "libcurand.so.10",
    "libcusolver.so.11",
    "libcusparse.so.12",
    "libcudnn.so.9",
    "libnccl.so.2",
)


def main() -> None:
    failures: list[str] = []
    for library in CUDA_LIBRARIES:
        try:
            ctypes.CDLL(library)
        except OSError as exc:
            failures.append(f"{library}: {exc}")
        else:
            print(f"{library}: OK")

    if failures:
        raise SystemExit("CUDA library failures:\n" + "\n".join(failures))

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("Physical GPUs:", gpus)
    if not gpus:
        raise SystemExit("TensorFlow did not detect a GPU")

    with tf.device("/GPU:0"):
        result = tf.matmul(tf.ones((1024, 1024)), tf.ones((1024, 1024)))
    print("Test operation device:", result.device)
    print("Test checksum:", float(tf.reduce_sum(result).numpy()))


if __name__ == "__main__":
    main()
