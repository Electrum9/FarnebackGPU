import sys
import numpy as np
import cv2
import ctypes
import os
import matplotlib.pyplot as plt
from time import time

# Load the shared library
lib = ctypes.cdll.LoadLibrary("./libframeproc.so")

# Setup function prototype
lib.process_frame.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input frame
    ctypes.POINTER(ctypes.c_float),  # output frame
    ctypes.c_int,                    # height
    ctypes.c_int,                    # width
]

def generate_gaussian_pyramid(image, num_levels):
    pyramid = [image.copy()]
    for i in range(num_levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def process_frame_with_cuda(frame):
    h, w = frame.shape
    frame_in = np.ascontiguousarray(frame, dtype=np.float32)
    num_levels = 4
    pyr_size = int(np.ceil(frame_in.size * ((1.0-(0.25**(num_levels+1)))/0.75)))
    frame_out = np.empty(pyr_size, dtype=np.float32)
    # frame_out = np.empty(4*frame.size, dtype=np.float32)

    lib.process_frame(
        frame_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        frame_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h, w
    )

    levels = []
    # levels.append(frame_out[:frame.size].reshape(h,w))
    # breakpoint()
    # levels.append(frame_out[frame.size:frame.size + (h+4)*(w+4)].reshape(h+4,w+4))

    # return levels
                  
    offset = 0

    for i in range(4):
        curr_level = frame_out[offset:offset + (frame.size >> (2*i))].reshape(h >> i, w >> i)
        print(f"{curr_level.shape=}")
        levels.append(curr_level)
        offset += curr_level.size
    return levels

def main(video):
    cap = cv2.VideoCapture(video)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (640,360))
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-9)
        print("starting processing")
        start = time()
        levels = process_frame_with_cuda(frame)
        # levels = generate_gaussian_pyramid(frame,4)
        end = time()
        print("done processing")
        print(f"{end-start=}")

        plt.imsave("frame.png", frame)
        for i, img in enumerate(levels):
            plt.imsave(f"level{i}.png", img)
        break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main("./walking.mp4")
