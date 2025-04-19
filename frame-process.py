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
    frame_out = np.empty_like(frame_in)

    lib.process_frame(
        frame_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        frame_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h, w
    )
    return frame_out

def main(video):
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-9)
        print("starting processing")
        processed = process_frame_with_cuda(frame)
        print("done processing")

        # Save or display
        plt.imsave("frame.png", processed)
        breakpoint()

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main("walking.mp4")
