#include <iostream>

#define TILE_ROWS 32 // 32 threads *
#define TILE_COLS 64 // 64 threads

/*
kernel size = 5, radius = 5//2 = 2
there are as many threads as there are output elements in the very final output (after the vertical pass)
assumption is that the image is already padded BEFORE passing into the convolution filter
*/
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
    }
}
