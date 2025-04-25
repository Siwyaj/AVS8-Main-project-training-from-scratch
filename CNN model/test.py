from ctypes import CDLL
CDLL(r"C:\tools\fluidsynth\bin\libfluidsynth-3.dll")

import tensorflow as tf

import tensorflow as tf
import torch

def check_tensorflow_cuda():
    print("TensorFlow CUDA Check:")
    if tf.config.list_physical_devices('GPU'):
        print("✅ TensorFlow can use GPU.")
    else:
        print("❌ TensorFlow can't detect CUDA.")

def check_pytorch_cuda():
    print("\nPyTorch CUDA Check:")
    if torch.cuda.is_available():
        print(f"✅ PyTorch can use GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ PyTorch can't detect CUDA.")

if __name__ == "__main__":
    check_tensorflow_cuda()
    check_pytorch_cuda()
