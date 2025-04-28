import sys
import numpy as np
import cv2
import ctypes
import os
import matplotlib.pyplot as plt

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
    frame_in = np.ascontiguousarray(frame, dtype=np.float32)
    num_levels = 4
    pyr_size = int(np.ceil(frame_in.size * ((1.0-(0.25**(num_levels+1)))/0.75)))
    frame_out = np.empty(pyr_size, dtype=np.float32)

    lib.process_frame(
        frame_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        frame_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h, w
    )

    levels = []
    offset = 0

    for i in range(4):
        curr_level = frame_out[offset:offset + (frame.size >> (2*i))].reshape(h >> i, w >> i)
        print(f"{curr_level.shape=}")
        levels.append(curr_level)
        offset += curr_level.size
    return levels

def main(video):
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-9)
        print("starting processing")
        levels = process_frame_with_cuda(frame)
        print("done processing")
        breakpoint()

        plt.imsave("frame.png", frame)
        for i, img in enumerate(levels):
            plt.imsave(f"level{i}.png", img)

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main("walking.mp4")
