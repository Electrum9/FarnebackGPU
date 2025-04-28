import sys
import numpy as np
import cv2
import ctypes
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time
# Load the shared library
lib = ctypes.cdll.LoadLibrary("./libframeproc.so")

# Setup function prototype
lib.process_frame.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input frame
    ctypes.POINTER(ctypes.c_float),  # output frame
    ctypes.c_int,                    # height
    ctypes.c_int,                    # width
]

def process_frame_with_cuda(frame):
    h, w = frame.shape
    pad = 2
    padded = np.pad(frame, ((pad,pad),(pad,pad)), mode='constant', constant_values= 0)
    h_pad, w_pad = padded.shape
    frame_in = np.ascontiguousarray(padded, dtype=np.float32)
    hooy = np.empty((int(h/2),int(w/2)), dtype=np.float32)
    boop = np.ascontiguousarray(hooy, dtype=np.float32)
    frame_out = np.empty_like(boop)

    lib.process_frame(
        frame_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        frame_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h_pad, w_pad
    )
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
    out = F.conv2d(frame, kernel_h, padding=(0, 2), stride=(1,2))
    out = F.conv2d(out, kernel_v, padding=(2, 0), stride=(2,1))
    print(f"PyTorch took {time.time() - start:.4f} seconds")
    return out.squeeze().cpu().numpy()


def main(video):
    cap = cv2.VideoCapture(video)
    frame_idx = 0 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        imp = np.zeros((64,64), dtype=np.float32)
        imp[0,0] = 1.0
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-9)
        print("starting processing")
        processed = process_frame_with_cuda(frame)
        #processed_impulse = process_frame_with_cuda(imp)
        print("done processing")
        torch_output = process_frame_with_python(frame, frame_idx)
        #torch_imp = process_frame_with_python(imp, frame_idx)
        abs_diff = np.abs(processed - torch_output)
        #abs_diff = np.abs(processed_impulse - torch_imp)
        max_diff = np.max(abs_diff)
        print(f"[Frame {frame_idx}] Max difference: {max_diff:.6f}")
        print(f"[Frame {frame_idx}] Mean difference: {abs_diff.mean():.6f}")
        frame_idx += 1
        np.set_printoptions(threshold=np.inf, linewidth=200, precision=4, suppress=True)
        # print("CUDA output (down-sampled):")
        # print(processed_impulse)
        # breakpoint()
        # print("\nPyTorch output (down-sampled):")
        # print(torch_imp)
        #difference = frame-processed 
        # Save or display
        plt.imsave("frame.png", processed)
        #plt.imsave("difference.png", difference)
        breakpoint()

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main("walking.mp4")