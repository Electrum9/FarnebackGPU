#pragma once

#define TILE_ROWS 32 // 32 threads *
#define TILE_COLS 32 // 64 threads
__constant__ float gaussianBlur_filter[5] = {1.0f/16, 4.0f/16, 6.0f/16, 4.0f/16, 1.0f/16};

template<int stride, int ksizeHalf>
__global__ void convolution1DKernel(
    float* input,
    const float* __restrict__ vFilter,
    const float* __restrict__ hFilter,
    float* output,
    int input_height,
    int input_width,
    int filter_size
);

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
 
    if (outx < input_width - 2*ksizeHalf && outy < input_height - 2*ksizeHalf) {
        output[outy * (input_width - 2*ksizeHalf) + outx] = result;
    }
 }

