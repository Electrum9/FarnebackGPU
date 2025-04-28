#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

__constant__ float gaussianBlur_filter[5] = {1.0f / 16, 4.0f / 16, 6.0f / 16,
                                             4.0f / 16, 1.0f / 16};

#define TILE_ROWS 32 // 32 threads
#define TILE_COLS 32 // 32 threads

template <int stride, int ksizeHalf>
__global__ void
conv2D_separable(float *input, const float *__restrict__ vFilter,
                    const float *__restrict__ hFilter, float *output,
                    int input_height, int input_width, int filter_size) {
  __shared__ float
      input_tile[TILE_ROWS * stride + 2 * ksizeHalf]
                [TILE_COLS * stride +
                 2 * ksizeHalf]; // to hold the input tile elements (+4 for the
                                 // halo region) --> formula comes from
                                 // reversing the conv output formula
  __shared__ float horizontal_tile[TILE_ROWS * stride + 2 * ksizeHalf]
                                  [TILE_COLS]; // to hold the elements after the
                                               // horizontal pass
  // final output is [TILE_ROWS][TILE_COLS]
  int tx = threadIdx.x; // column
  int ty = threadIdx.y; // row
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bdx = blockDim.x; // how many columns --> 64 (width)
  int bdy = blockDim.y; // how many rows --> 32 (height)

  int NUM_BANKS = 32;

  int outx = bx * bdx + tx; // x-coordinate of the output
  int outy = by * bdy + ty; // y-coordinate of the output

  // FILTERS, if kernel size is 5, load the guassianBlur_filter, else use input
  // vfilter and hfilter
  const float *vF = vFilter;
  const float *hF = hFilter;

  // LOADING IN THE DATA
  for (int row = ty; row < TILE_ROWS * stride + 2 * ksizeHalf; row += bdy) {
    for (int col = tx; col < TILE_COLS * stride + 2 * ksizeHalf; col += bdx) {
      int inputx =
          bx * bdx * stride + col - ksizeHalf; // to adjust for the halo region
      int inputy = by * bdy * stride + row - ksizeHalf;

      // check if data is within bounds
      if (inputx >= 0 && inputx < input_width && inputy >= 0 &&
          inputy < input_height) {
        input_tile[row][col] = input[inputy * input_width + inputx];
      }
    }
  }
  __syncthreads();

  // HORIZONTAL PASS
  for (int row = ty; row < TILE_ROWS * stride + 2 * ksizeHalf; row += bdy) {
    for (int col = tx; col < TILE_COLS; col += bdx) {
      float result = 0.0f;
      int hrow = row;
      int hcol = col * stride + ksizeHalf;

      // need to iterate from col-2 to col+2
      for (int k = -ksizeHalf; k <= ksizeHalf; ++k) {
        result += input_tile[hrow][hcol + k] *
                  hF[k + ksizeHalf]; // should it be hrow+ ksizehalf?
      }

      // want to write into shared memory with cyclic shifting
      int rotated_col =
          (col + row) % NUM_BANKS; // or should this be NUM_BANKS or TILE_COLS
      horizontal_tile[row][rotated_col] = result;
    }
  }
  __syncthreads();

  // VERTICAL PASS
  float result = 0.0f;
  int vrow = ty * stride + ksizeHalf;
  for (int k = -ksizeHalf; k <= ksizeHalf; ++k) {
    int row_offset = vrow + k;
    if (row_offset >= 0 && row_offset < TILE_ROWS * stride + 2 * ksizeHalf) {
      int rotated_col = (tx + row_offset) % NUM_BANKS;
      result += horizontal_tile[row_offset][rotated_col] * vF[k + ksizeHalf];
    }
  }

  output[outy * (input_width - 2 * ksizeHalf) + outx] = result;
}


void conv_gaussian(float *input, float *output, int input_height, int input_width) {
  dim3 threads(TILE_COLS, TILE_ROWS);
  dim3 blocks((input_width + TILE_COLS - 1) / TILE_COLS, 
              (input_height + TILE_ROWS - 1) / TILE_ROWS);
  conv2D_separable<2,2><<<blocks, threads>>>(input, gaussianBlur_filter, gaussianBlur_filter, output, input_height, input_width, 5);
    cudaDeviceSynchronize();
    printf("done gauss\n");
  //convolution1DKernel(input, output, input_height, input_width, 5, 2);
}

void gaussian_pyramid(float *output, int input_height, int input_width, int num_levels) {

  /*
     1. Allocate memory for pyramid
     2. Copy image to pyramid buffer (CPU to GPU)
     3. Compute Pyramid
  */

  // Generate remaining layers of pyramid via filtering and downsampling

  int curr_height = input_height;
  int curr_width = input_width;
  int running_size = 0;

  for (int i = 1; i < num_levels; i++) {

    int curr_size = curr_height*curr_width;
    conv_gaussian(output + running_size, output + running_size + curr_size, curr_height, curr_width);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    printf("level=%d\n", i);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    running_size += curr_size;
    curr_height >>= 2;
    curr_width >>= 2;
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
    cudaMemcpy(pyramid, input, img_size*sizeof(float), cudaMemcpyHostToDevice);
    //conv_gaussian(pyramid, pyramid + total, height, width);
    gaussian_pyramid(pyramid, height, width, num_levels);
    cudaMemcpy(output, pyramid, pyr_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(pyramid);
}

