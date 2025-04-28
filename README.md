# Farneback Algorithm on GPU

## Data

The sample video file used for development/experimentation is ``walking.mp4``,
sourced from [here](https://www.pexels.com/video/man-hiking-in-the-woods-3135811/).

## Video Loading and Processing

Video loading is currently done using the ``opencv-python`` library. The video frames
are then sent to a C wrapper which acts as an intermediary between Python and the CUDA kernel,
transferring results/data back and forth.

## To Run Code
There are 3 individual functions that we wrote and tested:

Convolution, Gaussian Pyramid, and Polynomial Expansion.

To test Convolution:
```
$ cd convolution
$ make process_frame
$ python frame-process.py
```
This should generate some images: frame.png, difference.png, pytorch_frame.png, and pytorch_difference.png. It will also print the timing and errors found between the CUDA and pytorch implementations of the functions for the different sizes of images.

To test the Polynomial Expansion:

```
$ cd Polynomial_Expansion
$ make poly_exp
$ python polyexp-Testing.py
```

This will print out the times for how long each kernel call takes for each size images. It also shows the error stats on momentum calculations, and matrices A,B,C for Polynomial Expansion.


To test the Gaussian Pyramid:

```
$ cd convolution
$ make process_gaussian
$ python frame-process-gpyr.py
```

This will print out the times for how long each kernel call takes for a given frame of the video. Additionally, it will save the images corresponding
to the different levels of the Gaussian pyramid (denoted as ``level0.png``, ``level1.png``, etc.)
