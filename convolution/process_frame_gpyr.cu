#include "convolution.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define TILE_ROWS 32 // 32 threads
#define TILE_COLS 32 // 32 threads

__global__ void zeroPad2D(const float* __restrict__ input, float* output,
                                     int input_rows, int input_cols,
                                     int output_rows, int output_cols,
                                     int pad_top, int pad_left) {
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_row >= output_rows || out_col >= output_cols)
        return;

    //output[out_row * output_cols + out_col] = 1.0f;

    int in_row = out_row - pad_top;
    int in_col = out_col - pad_left;

    if (in_row >= 0 && in_row < input_rows && in_col >= 0 && in_col < input_cols) {
        output[out_row * output_cols + out_col] = input[in_row * input_cols + in_col];
    }
}

void zero_pad(float *input, float *output, int h, int w, int pad_top, int pad_left) {
    int zp_height = h + 2*pad_top;
    int zp_width = w + 2*pad_left;
    printf("h=%d\n",h);
    printf("w=%d\n",w);

    dim3 blockDim(16, 16);
dim3 gridDim((zp_width + blockDim.x - 1) / blockDim.x,
             (zp_height + blockDim.y - 1) / blockDim.y);

    zeroPad2D<<<gridDim,blockDim>>>(input, output, h, w, zp_height, zp_width, pad_top, pad_left);
    cudaDeviceSynchronize();
}

void conv_gaussian(float *input, float *output, int h, int w) {
    int filter_size = 5;

    dim3 threads(TILE_COLS, TILE_ROWS);
    dim3 blocks(((w - 4) / 2 + TILE_COLS - 1) / TILE_COLS, 
                ((h - 4) / 2 + TILE_ROWS - 1) / TILE_ROWS);

    //conv_gaussian(output + running_size, output + running_size + curr_size, curr_height, curr_width);
    convolution1DKernel<2,2><<<blocks, threads>>>(input, nullptr, nullptr, output, h, w, filter_size);
    cudaDeviceSynchronize();
}

void gaussian_pyramid(float *output, int input_height, int input_width, int num_levels) {
  // Generate remaining layers of pyramid via filtering and downsampling

  int filter_size = 5;
  int prev_height = input_height;
  int prev_width = input_width;
  //int running_size = prev_height * prev_width;
  //int curr_size = (prev_height * prev_width) / 4;
  float *prev_img = output;
  float *curr_img = output + prev_height*prev_width;


  for (int i = 1; i < num_levels; i++) {
    float *prev_img_zp;

    int zp_height = prev_height + 4;
    int zp_width = prev_width + 4;
    int zp_size = zp_height*zp_width;
    cudaMalloc(&prev_img_zp, sizeof(float)*zp_size);

    zero_pad(prev_img, prev_img_zp, prev_height, prev_width, 2, 2);
    conv_gaussian(prev_img_zp, curr_img, prev_height+4, prev_width+4);

    cudaError_t err = cudaGetLastError();
    printf("level=%d\n", i);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    printf("prev_height=%d\n", prev_height);
    printf("prev_width=%d\n", prev_width);

    prev_height /= 2;
    prev_width /= 2;
    //running_size += prev_height * prev_width;
    prev_img = curr_img;
    curr_img = prev_img + prev_height*prev_width;
  }
}

extern "C"
void process_frame(float* input, float* output, int height, int width) {
    int total = height * width;
    printf("height=%d\n", height);
    printf("width=%d\n", width);
    printf("total=%d\n", total);

    int num_levels = 4;
    int img_size = height*width;
    int pyr_size = ceil(img_size * ((1.0-pow(0.25,num_levels+1))/0.75));
    printf("pyr_size=%d\n", pyr_size);

    float *pyramid;
    cudaMalloc(&pyramid, pyr_size*sizeof(float));
    cudaMemset(pyramid, 0.0f, pyr_size*sizeof(float));
    cudaMemcpy(pyramid, input, img_size*sizeof(float), cudaMemcpyHostToDevice);
    //conv_gaussian(pyramid, pyramid + total, height, width);

    //zero_pad(pyramid, pyramid+img_size, height, width, 2, 2);
    gaussian_pyramid(pyramid, height, width, num_levels);
    //cudaMemcpy(output, pyramid, pyr_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output, pyramid, pyr_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(pyramid);
}

