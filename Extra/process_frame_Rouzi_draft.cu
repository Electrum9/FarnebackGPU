#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define TILE_ROWS 32 // 32 threads
#define TILE_COLS 32 // 32 threads


#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y
#define bdx blockDim.x
#define bdy blockDim.y
__constant__ float gaussianBlur_filter[5] = {1.0f/16, 4.0f/16, 6.0f/16, 4.0f/16, 1.0f/16};
// 2*n+1 == 11 for polyN=5
__constant__ float c_g[11];
__constant__ float c_xg[11];
__constant__ float c_xxg[11];

// four Farnebäck inverses
__constant__ double c_ig03;
__constant__ double c_ig11;
__constant__ double c_ig33;
__constant__ double c_ig55;

// Function to choose whihc PolyExpansion filter to load from constant cache memory
__device__ __forceinline__   //inline so no compiling, just a nameholder to copy paste into where it's called
const float* PolyFilter(int id)
{
    // id == 0 → g   (Normal Gaussian)
    // id == 1 → xg  (first order Gaussian)
    // id == 2 → xxg (second order Gaussian)
    switch(id) {
      case 1: return c_xg;
      case 2: return c_xxg;
      default: return c_g;
    }
}

template<int stride, int ksizeHalf>
__global__ void convolution1DKernel(float* input, 
    int    vFilterId,    // 0=g, 1=xg, 2=xxg  (to avoid passing kernels, use const value filters choose with id)
    int    hFilterId,    // 0=g, 1=xg, 2=xxg  (to avoid passing kernels, use const value filters choose with id)
    float* output, int input_height, int input_width, int filter_size)
{
    __shared__ float input_tile[TILE_ROWS * stride + 2*ksizeHalf][TILE_COLS * stride + 2*ksizeHalf]; //to hold the input tile elements (+4 for the halo region) --> formula comes from reversing the conv output formula
    __shared__ float horizontal_tile[TILE_ROWS * stride + 2*ksizeHalf][TILE_COLS]; // to hold the elements after the horizontal pass
    // final output is [TILE_ROWS][TILE_COLS]
    int NUM_BANKS = 32;

    int outx = bx * bdx + tx; // x-coordinate of the output 
    int outy = by * bdy + ty; // y-coordinate of the output 

    // FILTERS, if kernel size is 5, load the guassianBlur_filter, 
    // else use input vfilter and hfilter id to load constant gaussian filters
    const float* vF = (filter_size == 5) ? gaussianBlur_filter : PolyFilter(vFilterId);
    const float* hF = (filter_size == 5) ? gaussianBlur_filter : PolyFilter(hFilterId);

    // LOADING IN THE DATA
    for (int row = ty; row < TILE_ROWS * stride + 2*ksizeHalf; row += bdy) {
        for (int col = tx; col < TILE_COLS * stride + 2*ksizeHalf; col += bdx){
            int inputx = bx * bdx * stride + col;
            int inputy = by * bdy * stride + row;
            
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
                result += input_tile[hrow][hcol + k] * hF[k+ksizeHalf]; 
            }
            // want to write into shared memory with cyclic shifting
            int rotated_col = (col + row) % NUM_BANKS; 
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
    // check if thread indices are withing frame bound, in case H,W arent dividable by Tile size
    if (outx < (input_width  - 2*ksizeHalf)/ stride && outy < (input_height - 2*ksizeHalf)/stride) 
    { 
        output[outy * (input_width - 2*ksizeHalf)/stride + outx] = result; // use oriignal width without padding!!
    }
}

extern "C"
void process_frame(float* input, float* output, int height, int width) {
    int total = height * width;
    int total_nopad = (height-4) * (width-4);
    float* d_in;
    float* d_out;
    printf("height=%d\n", height);
    printf("width=%d\n", width);
    printf("total=%d\n", total);

    cudaMalloc(&d_in, total * sizeof(float));
    cudaMalloc(&d_out, total_nopad/4 * sizeof(float));

    cudaMemcpy(d_in, input, total*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TILE_COLS, TILE_ROWS);
    dim3 blocks(((width - 4)/2 + TILE_COLS - 1) / TILE_COLS, 
                ((height - 4)/2+ TILE_ROWS - 1) / TILE_ROWS);
    
    // Use the convolution kernel instead of invert_colors
    int filter_size = 5; // Your filter is 5x5
    int stride = 2;      // Using stride of 1 as in your code
    convolution1DKernel<2,2><<<blocks, threads>>>(d_in, 0, 0, d_out, height, width, filter_size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(output, d_out, total_nopad/4*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

