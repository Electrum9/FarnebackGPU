import ctypes
import numpy as np
import cv2

# Load the shared library (adjust the path as needed)
lib = ctypes.cdll.LoadLibrary("./build/libframeproc.so")

# Set up function prototypes.

# process_frame: CUDA inversion
lib.process_frame.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # input frame pointer
    ctypes.POINTER(ctypes.c_ubyte),  # output frame pointer
    ctypes.c_int,                    # height
    ctypes.c_int,                    # width
    ctypes.c_int                     # channels
]

# process_optical_flow: CPU Farneback optical flow
lib.process_optical_flow.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # previous frame pointer (grayscale)
    ctypes.POINTER(ctypes.c_ubyte),  # current frame pointer (grayscale)
    ctypes.POINTER(ctypes.c_float),  # output flow pointer
    ctypes.c_int,                    # height
    ctypes.c_int                     # width
]

def process_frame_with_cuda(frame):
    # Inverts colors using CUDA.
    h, w, c = frame.shape
    frame_in = np.ascontiguousarray(frame, dtype=np.uint8)
    frame_out = np.empty_like(frame_in)
    lib.process_frame(
        frame_in.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        frame_out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        h, w, c
    )
    return frame_out

def process_optical_flow_wrapper(prev_gray, next_gray):
    # Processes two grayscale images (uint8) and returns the flow field (float32).
    h, w = prev_gray.shape
    prev_gray = np.ascontiguousarray(prev_gray, dtype=np.uint8)
    next_gray = np.ascontiguousarray(next_gray, dtype=np.uint8)
    flow = np.empty((h, w, 2), dtype=np.float32)
    lib.process_optical_flow(
        prev_gray.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        next_gray.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        flow.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h, w
    )
    return flow

# Example usage in a video processing script.
video_path = "walking.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

# Resize if needed.
resize_width, resize_height = 320, 180
prev_frame = cv2.resize(prev_frame, (resize_width, resize_height))
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# For output video (optical flow visualization)
fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
output_filename = "optical_flow_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_size = (prev_gray.shape[1], prev_gray.shape[0])
out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

# Create an HSV mask for visualization.
hsv_mask = np.zeros((resize_height, resize_width, 3), dtype=np.uint8)
hsv_mask[..., 1] = 255

print("Processing video with custom optical flow and CUDA frame processing...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (resize_width, resize_height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Optionally, use the CUDA color inversion function:
    inverted = process_frame_with_cuda(frame)
    # (You can display or save 'inverted' if desired.)
    
    # Compute optical flow between prev_gray and current gray frame using our CPU wrapper.
    flow = process_optical_flow_wrapper(prev_gray, gray)
    
    # Convert flow to HSV for visualization.
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    
    out.write(flow_bgr)
    cv2.imshow("Optical Flow", flow_bgr)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    prev_gray = gray.copy()

cap.release()
out.release()
cv2.destroyAllWindows()
