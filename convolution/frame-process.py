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

lib.process_frame_s2.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input frame
    ctypes.POINTER(ctypes.c_float),  # output frame
    ctypes.c_int,                    # height
    ctypes.c_int,                    # width
]

function_per_stride = {
    1: lib.process_frame,
    2: lib.process_frame_s2
}

def process_frame_with_cuda(frame, frame_idx, stride=1):
    h, w = frame.shape
    pad = 2
    padded = np.pad(frame, ((pad,pad),(pad,pad)), mode='constant', constant_values= 0)
    h_pad, w_pad = padded.shape
    frame_in = np.ascontiguousarray(padded, dtype=np.float32)
    # boop = np.ascontiguousarray(np.zeros((360, 640), dtype=np.float32)) 
    boop = np.ascontiguousarray(frame, dtype=np.float32)
    frame_out = np.empty_like(boop)
    
    print(f"Input dimensions: {frame_in.shape}")
    print(f"Output dimensions: {frame_out.shape}")
    print(f"[Frame {frame_idx}] CUDA version:")
    
    process_func = function_per_stride.get(stride)
    start = time.time()
    process_func(
        frame_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        frame_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h_pad, w_pad
    )
    print(f"    CUDA took {time.time() - start:.4f} seconds")
    return frame_out

def process_frame_with_python(frame_np, frame_idx, stride=1):
    gaussian_filter = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
    gaussian_filter /= gaussian_filter.sum()
    
    frame = torch.tensor(frame_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frame = frame.to(device)
    kernel = gaussian_filter.to(device)
    
    kernel_h = kernel.view(1, 1, 1, -1)
    kernel_v = kernel.view(1, 1, -1, 1)
    
  
    pad = 2
    
    print(f"[Frame {frame_idx}] Python version:")
    start = time.time()
    out = F.conv2d(frame, kernel_h, padding=(0, pad))
    out = F.conv2d(out, kernel_v, padding=(pad, 0))
    
    print(f"    PyTorch took {time.time() - start:.4f} seconds")
    return out.squeeze().cpu().numpy()


def large_image_s1(video, stride=1): #1280x720
    cap = cv2.VideoCapture(video)
    frame_idx = 0
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-9)

    # RUNNING CUDA VERSION
    cuda_output = process_frame_with_cuda(frame, frame_idx, stride)
    difference = frame-cuda_output
    plt.imsave("frame.png", cuda_output)
    plt.imsave("difference.png", difference)
    
    # RUNNING PYTORCH VERSION
    torch_output = process_frame_with_python(frame, frame_idx, stride)
    different = frame - torch_output
    plt.imsave("pytorch_frame.png", torch_output)
    plt.imsave("pytorch_difference.png", difference)
    
    # COMPARE THE TWO 
    abs_diff = np.abs(cuda_output - torch_output)
    max_diff = np.max(abs_diff)
    print(f"[Frame {frame_idx}] Max difference: {max_diff:.6f}")
    print(f"[Frame {frame_idx}] Mean difference: {abs_diff.mean():.6f}")

    cap.release()
    cv2.destroyAllWindows()
    
def medium_image_s1(video, stride=1): #640x360
    cap = cv2.VideoCapture(video)
    frame_idx = 0
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-9)
    # frame = cv2.resize(frame, (644, 364))
    frame = cv2.resize(frame, (640, 360))
    
    # RUNNING CUDA VERSION
    
    cuda_output = process_frame_with_cuda(frame, frame_idx, stride)
    difference = frame-cuda_output
    plt.imsave("frame.png", cuda_output)
    plt.imsave("difference.png", difference)
    
    # RUNNING PYTORCH VERSION
    torch_output = process_frame_with_python(frame, frame_idx, stride)
    different = frame - torch_output
    plt.imsave("pytorch_frame.png", torch_output)
    plt.imsave("pytorch_difference.png", difference)
    
    # COMPARE THE TWO 
    abs_diff = np.abs(cuda_output - torch_output)
    max_diff = np.max(abs_diff)
    print(f"[Frame {frame_idx}] Max difference: {max_diff:.6f}")
    print(f"[Frame {frame_idx}] Mean difference: {abs_diff.mean():.6f}")

    cap.release()
    cv2.destroyAllWindows()
    
def smallest_image_s1(video, stride=1): #320, 180
    cap = cv2.VideoCapture(video)
    frame_idx = 0
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-9)
    frame = cv2.resize(frame, (320, 180))
    
    # RUNNING CUDA VERSION
    cuda_output = process_frame_with_cuda(frame, frame_idx, stride)
    difference = frame-cuda_output
    plt.imsave("frame.png", cuda_output)
    plt.imsave("difference.png", difference)
    
    # RUNNING PYTORCH VERSION
    torch_output = process_frame_with_python(frame, frame_idx, stride)
    different = frame - torch_output
    plt.imsave("pytorch_frame.png", torch_output)
    plt.imsave("pytorch_difference.png", difference)
    
    # COMPARE THE TWO 
    abs_diff = np.abs(cuda_output - torch_output)
    max_diff = np.max(abs_diff)
    print(f"[Frame {frame_idx}] Max difference: {max_diff:.6f}")
    print(f"[Frame {frame_idx}] Mean difference: {abs_diff.mean():.6f}")

    cap.release()
    cv2.destroyAllWindows()
    
# def smallest_image(video): #320, 180
#     cap = cv2.VideoCapture(video)
#     frame_idx = 0
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-9)
#         frame = cv2.resize(frame, (324, 184))
        
#         # RUNNING CUDA VERSION
        
#         cuda_output = process_frame_with_cuda(frame, frame_idx)
#         difference = frame-cuda_output
#         plt.imsave("frame.png", cuda_output)
#         plt.imsave("difference.png", difference)
        
#         # RUNNING PYTORCH VERSION
#         torch_output = process_frame_with_python(frame, frame_idx)
#         different = frame - torch_output
#         plt.imsave("pytorch_frame.png", torch_output)
#         plt.imsave("pytorch_difference.png", difference)
        
#         # COMPARE THE TWO 
#         abs_diff = np.abs(cuda_output - torch_output)
#         max_diff = np.max(abs_diff)
#         print(f"[Frame {frame_idx}] Max difference: {max_diff:.6f}")
#         print(f"[Frame {frame_idx}] Mean difference: {abs_diff.mean():.6f}")

#     cap.release()
#     cv2.destroyAllWindows()

def main(video):
    print(f"For one frame, with stride = 1, \n")
    print("SMALL IMAGE:")
    smallest_image_s1(video, stride=1)
    print("\nMEDIUM IMAGE")
    medium_image_s1(video, stride=1)
    print("\nLARGE IMAGE")
    large_image_s1(video, stride=1)

    

if __name__=="__main__":
    main("../walking.mp4")