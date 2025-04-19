#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define TILE_ROWS 32 // 32 threads *
#define TILE_COLS 32 // 64 threads

__global__ void invert_colors(float* input, float* output, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = height * width;
    if (idx < total) {
        output[idx] = 1 - input[idx];  // Simple invert
    }
}

__global__ void convolution1DKernel(float* input, float* output, int input_height, int input_width, int filter_size, int stride)
{
    // __shared__ float input_tile[TILE_ROWS * stride + 4][TILE_COLS * stride + 4]; //to hold the input tile elements (+4 for the halo region) --> formula comes from reversing the conv output formula
    // __shared__ float horizontal_tile[TILE_ROWS * stride + 4][TILE_COLS]; // to hold the elements after the horizontal pass

    __shared__ float input_tile[TILE_ROWS * 1 + 4][TILE_COLS * 1 + 4]; //to hold the input tile elements (+4 for the halo region) --> formula comes from reversing the conv output formula
    __shared__ float horizontal_tile[TILE_ROWS * 1 + 4][TILE_COLS]; // to hold the elements after the horizontal pass
    // final output is [TILE_ROWS][TILE_COLS]

    int tx = threadIdx.x; // column
    int ty =  threadIdx.y; //row
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bdx = blockDim.x; //how many columns --> 64 (width)
    int bdy = blockDim.y;//how many rows --> 32 (height)

    int NUM_BANKS = 32;

    int outx = bx * bdx + tx; // x-coordinate of the output
    int outy = by * bdy + ty; // y-coordinate of the output

    // FILTERS
    float filter[5] = {1.0f/16, 4.0f/16, 6.0f/16, 4.0f/16, 1.0f/16};
    // float filter[5] = {1.0f, 4.0f, 6.0f, 4.0f, 1.0f};

    // LOADING IN THE DATA
    for (int row = ty; row < TILE_ROWS * stride + 4; row += bdy) {
        for (int col = tx; col < TILE_COLS * stride + 4; col += bdx){
            int inputx = bx * bdx * stride + col - 2; // to adjust for the halo region
            int inputy = by * bdy * stride + row - 2;

            // check if data is within bounds 
            if (inputx >= 0 && inputx < input_width && inputy >= 0 && inputy < input_height){
                input_tile[row][col] = input[inputy * input_width + inputx];
            }
        }
    }
    __syncthreads();

    // HORIZONTAL PASS
    for (int row=ty; row < TILE_ROWS * stride + 4; row += bdy){
        for (int col = tx; col < TILE_COLS; col += bdx){
            float result = 0.0f;
            int hrow = row; // or should this be normal row
            int hcol = col * stride + 2;

            // need to iterate from col-2 to col+2
            for (int k = -2; k <= 2; ++ k){
                result += input_tile[hrow][hcol + k] * filter[k+2];
            }

            // want to write into shared memory with cyclic shifting
            int rotated_col = (col + row) % NUM_BANKS; // or should this be NUM_BANKS or TILE_COLS
            horizontal_tile[row][rotated_col] = result;
        }
    }
    __syncthreads();

    // VERTICAL PASS
    for (int col = tx; col < TILE_COLS; col += bdy){
        float result = 0.0f;
        for (int row = ty; row < TILE_ROWS; row += bdx){
            int vrow = row * stride + 2;
            for (int k = -2; k<= 2; ++k){
                int row_offset = vrow + k;
                if (row_offset >= 0 && row_offset < TILE_ROWS * stride + 4) {
                    int rotated_col = (col + row_offset) % NUM_BANKS;
                    result += horizontal_tile[row_offset][rotated_col] * filter[k + 2];
                }
            }
        }
        output[outy * input_width + outx] = result;

        // Debug print
        // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("Running kernel! block=(%d, %d), thread=(%d, %d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
        // }

        // if (outy < 2 && outx < 2) {
        //     printf("Thread (%d, %d) => output[%d][%d] = %f\n", 
        //         outy, outx, outy, outx, result);
        // }

    }
}

extern "C"
void process_frame(float* input, float* output, int height, int width) {
    int total = height * width;
    float* d_in;
    float* d_out;
    printf("height=%d\n", height);
    printf("width=%d\n", width);
    printf("total=%d\n", total);

    cudaMalloc(&d_in, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));

    cudaMemcpy(d_in, input, total*sizeof(float), cudaMemcpyHostToDevice);

    // printf("First few input values:\n");
    // for (int i = 0; i < 10 && i < total; ++i) {
    //     printf("input[%d] = %f\n", i, input[i]);
    // }

    dim3 threads(TILE_COLS, TILE_ROWS);
    dim3 blocks((width + TILE_COLS - 1) / TILE_COLS, 
                (height + TILE_ROWS - 1) / TILE_ROWS);
    
    // Use the convolution kernel instead of invert_colors
    int filter_size = 5; // Your filter is 5x5
    int stride = 1;      // Using stride of 1 as in your code
    convolution1DKernel<<<blocks, threads>>>(d_in, d_out, height, width, filter_size, stride);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(output, d_out, total*sizeof(float), cudaMemcpyDeviceToHost);

    // printf("First few output values:\n");
    // for (int i = 0; i < 10 && i < total; ++i) {
    //     printf("output[%d] = %f\n", i, output[i]);
    // }

    cudaFree(d_in);
    cudaFree(d_out);
}

