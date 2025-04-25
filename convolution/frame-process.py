import sys
import numpy as np
import cv2
import ctypes
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Load the shared library
lib = ctypes.cdll.LoadLibrary("./libframeproc.so")

# Setup function prototype
lib.process_frame.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input frame
    ctypes.POINTER(ctypes.c_float),  # output frame
    ctypes.c_int,                    # height
    ctypes.c_int,                    # width
]

def process_frame_with_cuda(frame, frame_idx, padding=0):
    frame = np.pad(frame, pad_width=padding, mode='edge')
    h, w = frame.shape
    frame_in = np.ascontiguousarray(frame, dtype=np.float32)
    frame_out = np.empty_like(frame_in)

    print(f"\n[Frame {frame_idx}] CUDA version:")
    start = time.time()
    lib.process_frame(
        frame_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        frame_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h, w
    )
    print(f"CUDA took {time.time() - start:.4f} seconds")
    return frame_out

def process_frame_with_python(frame_np, frame_idx):
    gaussian_filter = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
    gaussian_filter /= gaussian_filter.sum()
    
    frame = torch.tensor(frame_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frame = frame.to(device)
    kernel = gaussian_filter.to(device)
    
    kernel_h = kernel.view(1, 1, 1, -1)
    kernel_v = kernel.view(1, 1, -1, 1)
    
    print(f"\n[Frame {frame_idx}] CUDA version:")
    start = time.time()
    out = F.conv2d(frame, kernel_h, padding=(0, 2))
    out = F.conv2d(out, kernel_v, padding=(2, 0))
    print(f"PyTorch took {time.time() - start:.4f} seconds")
    return out.squeeze().cpu().numpy()

def main(video):
    cap = cv2.VideoCapture(video)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-9)
        
        # RUNNING CUDA VERSION
        
        cuda_output = process_frame_with_cuda(frame, frame_idx)
        difference = frame-cuda_output
        plt.imsave("frame.png", cuda_output)
        plt.imsave("difference.png", difference)
        
        # RUNNING PYTORCH VERSION
        torch_output = process_frame_with_python(frame, frame_idx)
        different = frame - torch_output
        plt.imsave("pytorch_frame.png", torch_output)
        plt.imsave("pytorch_difference.png", difference)
        
        # COMPARE THE TWO 
        abs_diff = np.abs(cuda_output - torch_output)
        max_diff = np.max(abs_diff)
        print(f"[Frame {frame_idx}] Max difference: {max_diff:.6f}")
        print(f"[Frame {frame_idx}] Mean difference: {abs_diff.mean():.6f}")
        frame_idx += 1
        breakpoint()

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main("../walking.mp4")