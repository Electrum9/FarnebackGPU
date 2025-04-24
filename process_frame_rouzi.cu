#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>


#define TILE_ROWS 32 // 32 threads *
#define TILE_COLS 32 // 64 threads
__constant__ float gaussianBlur_filter[5] = {1.0f/16, 4.0f/16, 6.0f/16, 4.0f/16, 1.0f/16};
/*
kernel size = 5, radius = 5//2 = 2
there are as many threads as there are output elements in the very final output (after the vertical pass)
assumption is that the image is already padded BEFORE passing into the convolution filter
*/
template<int stride, int ksizeHalf>
__global__ void convolution1DKernel(float* input, const float* __restrict__ vFilter,
    const float* __restrict__ hFilter, float* output, int input_height, int input_width, int filter_size)
{
    // __shared__ float input_tile[TILE_ROWS * stride + 4][TILE_COLS * stride + 4]; //to hold the input tile elements (+4 for the halo region) --> formula comes from reversing the conv output formula
    // __shared__ float horizontal_tile[TILE_ROWS * stride + 4][TILE_COLS]; // to hold the elements after the horizontal pass
    //const int ksizeHalf = filter_size / /2
    __shared__ float input_tile[TILE_ROWS * stride + 2*ksizeHalf][TILE_COLS * stride + 2*ksizeHalf]; //to hold the input tile elements (+4 for the halo region) --> formula comes from reversing the conv output formula
    __shared__ float horizontal_tile[TILE_ROWS * stride + 2*ksizeHalf][TILE_COLS]; // to hold the elements after the horizontal pass
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

    // FILTERS, if kernel size is 5, load the guassianBlur_filter, else use input vfilter and hfilter
    const float* vF = (filter_size == 5) ? gaussianBlur_filter : vFilter;
    const float* hF = (filter_size == 5) ? gaussianBlur_filter : hFilter;

    // LOADING IN THE DATA
    for (int row = ty; row < TILE_ROWS * stride + 2*ksizeHalf; row += bdy) {
        for (int col = tx; col < TILE_COLS * stride + 2*ksizeHalf; col += bdx){
            int inputx = bx * bdx * stride + col - ksizeHalf; // to adjust for the halo region
            int inputy = by * bdy * stride + row - ksizeHalf;

            // check if data is within bounds 
            if (inputx >= 0 && inputx < input_width && inputy >= 0 && inputy < input_height){
                input_tile[row][col] = input[inputy * input_width + inputx];
            }
        }
    }
    __syncthreads();

    // HORIZONTAL PASS
    for (int row=ty; row < TILE_ROWS * stride + 2*ksizeHalf; row += bdy){
        for (int col = tx; col < TILE_COLS; col += bdx){
            float result = 0.0f;
            int hrow = row; 
            int hcol = col * stride + ksizeHalf;

            // need to iterate from col-2 to col+2
            for (int k = -ksizeHalf; k <= ksizeHalf; ++ k){
                result += input_tile[hrow][hcol + k] * hF[k+ksizeHalf];  // should it be hrow+ ksizehalf?
            }

            // want to write into shared memory with cyclic shifting
            int rotated_col = (col + row) % NUM_BANKS; // or should this be NUM_BANKS or TILE_COLS
            horizontal_tile[row][rotated_col] = result;
        }
    }
    __syncthreads();

    // VERTICAL PASS
    float result = 0.0f;
    int vrow = ty * stride + ksizeHalf;
    for (int k = -ksizeHalf; k<= ksizeHalf; ++k){
        int row_offset = vrow + k;
        if (row_offset >= 0 && row_offset < TILE_ROWS * stride + 2*ksizeHalf) {
            int rotated_col = (tx + row_offset) % NUM_BANKS;
            result += horizontal_tile[row_offset][rotated_col] * vF[k + ksizeHalf];
        }
    }

    output[outy * input_width + outx] = result;
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
    //int halfK = filter_size/2;
     // Using stride of 1 as in your code
    #define GAUS_KERNEL_SIZE 5
    #define POLY_KERNEL_SIZE 11
    #define HALF_GAUS_KERNEL_SIZE 2
    #define HALF_POLY_KERNEL_SIZE 5
    convolution1DKernel<1,HALF_GAUS_KERNEL_SIZE><<<blocks, threads>>>(d_in, nullptr, nullptr, d_out, height, width, filter_size);
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
