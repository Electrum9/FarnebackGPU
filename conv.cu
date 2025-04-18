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
    __shared__ float input_tile[TILE_ROWS * stride + 4][TILE_COLS * stride + 4]; //to hold the input tile elements (+4 for the halo region) --> formula comes from reversing the conv output formula
    __shared__ float horizontal_tile[TILE_ROWS * stride + 4][TILE_COLS]; // to hold the elements after the horizontal pass
    // final output is [TILE_ROWS][TILE_COLS]

    int tx = threadIdx.x // column
    int ty =  threadIdx.y //row
    #define bx blockIdx.x
    #define by blockIdx.y
    #define bdx blockDim.x //how many columns --> 64 (width)
    #define bdy blockDim.y //how many rows --> 32 (height)

    int outx = bx * blockDim.x + tx; // x-coordinate of the output
    int outy = by * blockDim.y + ty; // y-coordinate of the output

    // FILTERS
    float filter[5] = {1.0f/16, 4.0f/16, 6.0f/16, 4.0f/16, 1.0f/16};

    // LOADING IN THE DATA
    for (int row = ty; row < TILE_ROWS * stride + 4; row += bdx) {
        for (int col = tx; col < TILE_COLS * stride + 4; col += bdy){
            int inputx = outx + col - 2; // to adjust for the halo region
            int inputy = outy + row - 2;

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
            int hrow = row * stride + 2; // or should this be normal row
            int hcol = col * stride + 2;

            // need to iterate from col-2 to col+2
            for (int k = -2; k <= 2; ++ k){
                result += input_tile[hrow][hcol + k] * filter[k+2];
            }

            // want to write into shared memory with cyclic shifting
            int rotated_col = (col + row) % TILE_COLS; // or should this be NUM_BANKS
            horizontal_tile[row][rotated_col] = result;
        }
    }
    __syncthreads();

    // VERTICAL PASS
    for (int col = tx; col < TILE_COLS; col += bdx){
        for (int row = ty; row < TILE_ROWS; row += bdy){
            float result = 0.0f;

            for (int k = -2; k<= 2; ++k){
                int row_offset = row + k;
                if (row_offset >= 0 && row_offset < TILE_ROWS * stride + 4) {
                    int rotated_col = (col + row_offset) % TILE_COLS;
                    result += horizontal_tile[row_offset][rotated_col] * filter[k + 2];
                }
            }
        }

        output[outy * input_width + outx] = result;
    }
}

int main()
{

    float input[7][7] = {
        {0, 0, 0, 0, 0, 0, 0},
        {0, 1, 2, 3, 2, 1, 0},
        {0, 4, 5, 6, 5, 4, 0},
        {0, 7, 8, 9, 8, 7, 0},
        {0, 4, 5, 6, 5, 4, 0},
        {0, 1, 2, 3, 2, 1, 0},
        {0, 0, 0, 0, 0, 0, 0}
    };

    int input_height = 7;
    int input_width = 7;
    int filter_size = 5;
    int stride = 1;

    float *d_input, *d_filter, *d_output;
    float output[7][7]; // Output buffer

    // Allocate device memory
    cudaMalloc(&d_input, sizeof(float) * input_height * input_width);
    cudaMalloc(&d_filter, sizeof(float) * filter_size);
    cudaMalloc(&d_output, sizeof(float) * input_height * input_width);

    // Copy input and filter to device memory
    cudaMemcpy(d_input, input, sizeof(float) * input_height * input_width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, sizeof(float) * filter_size, cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 threads_per_block(16, 16); // You can adjust this based on your hardware
    dim3 num_blocks(1, 1);  // You can adjust this based on the image size
    convolution1DKernel<<<num_blocks, threads_per_block>>>(d_input, d_filter, d_output,
                                                           input_height, input_width,
                                                           filter_size, stride);

    // Copy the result back to host memory
    cudaMemcpy(output, d_output, sizeof(float) * input_height * input_width, cudaMemcpyDeviceToHost);

    // Print the output (for verification)
    std::cout << "Final Output from CUDA:\n";
    for (int i = 2; i < 5; ++i) {
        for (int j = 2; j < 5; ++j) {
            std::cout << output[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    return 0;
}

