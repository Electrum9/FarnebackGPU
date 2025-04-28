#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "convolution.cuh"

#define TILE_ROWS 32 // 32 threads *
#define TILE_COLS 32 // 64 threads
// __constant__ float gaussianBlur_filter[5] = {1.0f/16, 4.0f/16, 6.0f/16, 4.0f/16, 1.0f/16};

__global__ void invert_colors(float* input, float* output, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = height * width;
    if (idx < total) {
        output[idx] = 1 - input[idx];  // Simple invert
    }
}

extern "C"
void process_frame(float* input, float* output, int height, int width) {
    int total = height * width;
    int total_nopad = (height - 4) * (width - 4);
    float* d_in;
    float* d_out;
    printf("height=%d\n", height);
    printf("width=%d\n", width);
    printf("total=%d\n", total);

    cudaMalloc(&d_in, total * sizeof(float));
    cudaMalloc(&d_out, total_nopad * sizeof(float));

    cudaMemcpy(d_in, input, total*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TILE_COLS, TILE_ROWS);

    // for a pixel at the edge, the convolution filter would n
    dim3 blocks((width - 4 + TILE_COLS - 1) / TILE_COLS, 
                (height - 4 + TILE_ROWS - 1) / TILE_ROWS);
    
    // Use the convolution kernel instead of invert_colors
    int filter_size = 5; // Your filter is 5x5
    convolution1DKernel<1,2><<<blocks, threads>>>(d_in, nullptr, nullptr, d_out, height, width, filter_size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return;
    }

    err = cudaMemcpy(output, d_out, total_nopad*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA error copying to output: %s\n", cudaGetErrorString(err));
    }
    
}

