import cv2
import numpy as np
import time

# Path to the saved video file (update this)
video_path = "walking.mp4"

# Set desired resolution (None to keep original size)
resize_width = 320  # Set to None to keep original width
resize_height = 180  # Set to None to keep original height

# Load the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

# Resize frame if needed
if resize_width and resize_height:
    prev_frame = cv2.resize(prev_frame, (resize_width, resize_height))

# Convert to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

# Define the output video writer
output_filename = "optical_flow_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_size = (prev_gray.shape[1], prev_gray.shape[0])  # (width, height)
out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

# Create a mask for visualization
hsv_mask = np.zeros_like(prev_frame)
hsv_mask[..., 1] = 255  # Set saturation to max

total_time = 0
frame_count = 0

print("Processing video with optical flow...")
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Resize frame if needed
    if resize_width and resize_height:
        frame = cv2.resize(frame, (resize_width, resize_height))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Start timing
    start_tick = cv2.getTickCount()

    # Compute optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    # End timing
    end_tick = cv2.getTickCount()
    elapsed_time = (end_tick - start_tick) / cv2.getTickFrequency()  # Convert to seconds
    total_time += elapsed_time
    frame_count += 1

    # Convert flow to HSV representation
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to BGR for visualization
    flow_bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    # Write frame to output video
    out.write(flow_bgr)

    prev_gray = gray.copy()

# Release resources
cap.release()
out.release()

# Print performance results
avg_time_per_frame = total_time / frame_count if frame_count > 0 else 0
print(f"Processed {frame_count} frames")
print(f"Total Optical Flow Computation Time: {total_time:.4f} sec")
print(f"Average Time per Frame: {avg_time_per_frame:.4f} sec")
print(f"Optical flow video saved as: {output_filename}")

