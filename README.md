# Farneback Algorithm on GPU

## Data

The sample video file used for development/experimentation is ``walking.mp4``,
sourced from [here](https://www.pexels.com/video/man-hiking-in-the-woods-3135811/).

## Video Loading and Processing

Video loading is currently done using the ``opencv-python`` library. The video frames
are then sent to a C wrapper which acts as an intermediary between Python and the CUDA kernel,
transferring results/data back and forth.
