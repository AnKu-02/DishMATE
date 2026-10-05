"""Inspect local detection dependencies without connecting to the robot."""
import sys
import torch
import ultralytics

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("CPU threads:", torch.get_num_threads())
