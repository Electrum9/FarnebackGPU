import numpy as np
import cv2
import ctypes
import os
import matplotlib.pyplot as plt

# Load the shared library
lib = ctypes.cdll.LoadLibrary("./libframeproc.so")

# Setup function prototype
lib.process_frame.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # input frame
    ctypes.POINTER(ctypes.c_ubyte),  # output frame
    ctypes.c_int,                    # height
    ctypes.c_int,                    # width
    ctypes.c_int                     # channels
]

def process_frame_with_cuda(frame):
    h, w, c = frame.shape
    frame_in = np.ascontiguousarray(frame, dtype=np.uint8)
    frame_out = np.empty_like(frame_in)

    lib.process_frame(
        frame_in.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        frame_out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        h, w, c
    )
    return frame_out

cap = cv2.VideoCapture("walking.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    print("starting processing")
    processed = process_frame_with_cuda(frame)
    print("done processing")

    # Save or display
    plt.imsave("frame.png", processed)

cap.release()
cv2.destroyAllWindows()

