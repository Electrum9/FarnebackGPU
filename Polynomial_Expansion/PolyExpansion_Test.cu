#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <cmath>
#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y
#define bdx blockDim.x
#define bdy blockDim.y
#define POLY_KERNEL_SIZE 11
#define HALF_POLY_KERNEL_SIZE 5
#define TILE_ROWS 32 // 32 threads
#define TILE_COLS 32 // 32 threads
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
    if (outx < (input_width  - 2*ksizeHalf) && outy < (input_height - 2*ksizeHalf)) 
    { 
    output[outy * (input_width - 2*ksizeHalf) + outx] = result; // use oriignal width without padding!!
    }
}


// this is on CPU host beforee launching the kernels, similar to OpenCV computation, but simpler/Faster
__host__ void prepareGaussian(int n, double sigma, std::vector<float>& g, std::vector<float>& xg, std::vector<float>& xxg,double& ig11, double& ig03, double& ig33, double& ig55)
{
    if (sigma < 1e-6) sigma = n*0.3;
    g.resize(2*n+1);
    xg.resize(2*n+1);
    xxg.resize(2*n+1);
    double s = 0;
    for(int i=-n;i<=n;i++){
      g[i+n] = std::exp(-i*i/(2*sigma*sigma));
      s += g[i+n];
    }
    s = 1./s;
    for(int i=-n;i<=n;i++){
      g [i+n] *= float(s);
      xg[i+n]  = i * g[i+n];
      xxg[i+n] = i*i * g[i+n];
    }
    // no need to build moment matrix G, and calculate the inverse ig03, ig11, ig33, ig55 later
    // invG looks like belo, can calcualte the scalar inverse elemetns with closed form formulation
    // // invG:
    // // [ x        e  e    ]
    // // [    y             ]
    // // [       y          ]
    // // [ e        z       ]
    // // [ e           z    ]
    // // [                u ]
    ig03 = 0;
    ig11 = 0;
    ig33 = 0;
    ig55 = 0;
    for(int i=-n; i<=n; ++i){
    double gi = g[i+n];
    ig03  +=   gi*   gi;        // ∑ g(x)^2
    ig11  += (i*gi)*(i*gi);     // ∑ [x·g(x)]^2  = ∑ x²·g(x)²
    ig55  += (i*i*gi)*(i*i*gi); // ∑ [x²·g(x)]^2 = ∑ x⁴·g(x)²
    ig33= ig11;                 // same closed‐form in Farnebäck paper
    }
}


#define cudaSafeCall(call) do {                                      \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
               cudaGetErrorString(err));                              \
        exit(err);                                                    \
    }                                                                 \
} while(0)

 // it's being called to copy the results of the preapreGaussian int constnat memories in the GPU
 __host__ void setPolynomialExpansionConsts( int polyN, const float *g, const float *xg, const float *xxg, double ig11, double ig03, double ig33, double ig55)
{
    cudaSafeCall(cudaMemcpyToSymbol(c_g, g, (2*polyN + 1) * sizeof(*g)));
    cudaSafeCall(cudaMemcpyToSymbol(c_xg, xg, (2*polyN + 1) * sizeof(*xg)));
    cudaSafeCall(cudaMemcpyToSymbol(c_xxg, xxg, (2*polyN + 1) * sizeof(*xxg)));
    cudaSafeCall(cudaMemcpyToSymbol(c_ig11, &ig11, sizeof(ig11)));
    cudaSafeCall(cudaMemcpyToSymbol(c_ig03, &ig03, sizeof(ig03)));
    cudaSafeCall(cudaMemcpyToSymbol(c_ig33, &ig33, sizeof(ig33)));
    cudaSafeCall(cudaMemcpyToSymbol(c_ig55, &ig55, sizeof(ig55)));
}


__global__ void computePolyCoefficients( float* d_Ix, float* d_Iy, float* d_Ixx, float* d_Iyy, float* d_Ixy, float* d_C, double* d_A, double* d_B, double* d_c, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    // let for each output pixel (wdith_nopad*height_nopad) one thread calculates matrix A, B, C
    int idx = y * width + x;
    // A matrix (2x2) per pixel
    float Axy =  0.5 * c_ig55 * d_Ixy[idx];
    d_A[4 * idx + 0] = c_ig03 * d_C[idx] + c_ig33 * d_Ixx[idx];  //Axx
    d_A[4 * idx + 1] = Axy; //Axy = Ayx
    d_A[4 * idx + 2] = Axy; //Ayx = Axy
    d_A[4 * idx + 3] = c_ig03 * d_C[idx] + c_ig33 * d_Iyy[idx]; //Ayy

    // B vector (2x1) per pixel
    d_B[2 * idx + 0] = d_Ix[idx] * c_ig11;  //Bx
    d_B[2 * idx + 1] = d_Iy[idx] * c_ig11;  //By

    // Scalar c (1x1) per pixel
    d_c[idx] = d_C[idx] * c_ig03;  //smoothed Gaussian image scaled
}

extern "C"
void polynomialExpansion(const float* d_src, float sigma, int width, int height, int polyN, double* outC, double* outB,double* outA, float* outdc, float* outIx, float* outIy, float* outIxx, float* outIyy, float* outIxy)
 // will call convolution1D kernel it's from host CPU, launches 6 stream to run kernels for conv1D
 {  
    // Get the Polynomial Expansion Basis Gaussian filters 
    int total = height * width;  // include the padding
    int total_nopad = (height - 2*polyN) * (width - 2*polyN); //calcualte not padded
    std::vector<float> g, xg, xxg;  //to hold Guassian kernel, first order, adn second order guassian
    double ig11, ig03, ig33, ig55;  //to hold inverse elements of matrix G

    prepareGaussian(polyN, sigma, g, xg, xxg, ig11, ig03, ig33, ig55);
    // make all these weights and kernels as global constant in GPU cache -> fast memory access, no read write to GPU kernel
    setPolynomialExpansionConsts(polyN, g.data(), xg.data(), xxg.data(), ig11, ig03, ig33, ig55);
    // Create 6 streams 
    cudaStream_t streams[6];
    for(int i = 0; i < 6; ++i)
        cudaStreamCreate(&streams[i]);

    // For input frame 
    float* d_in;
    cudaMalloc(&d_in, total * sizeof(float));
    cudaMemcpy(d_in, d_src, total*sizeof(float), cudaMemcpyHostToDevice);

    // allocate six device buffers for the 6 raw responses to output for correctness check against PyTorch
    // r1, r2, .....r6 = Ic, Ix, Iy, Ixx, Ixy, Iyy
    float *d_C, *d_Ix, *d_Iy, *d_Ixx, *d_Iyy, *d_Ixy;
    cudaMalloc(&d_C,   total_nopad*sizeof(float));
    cudaMalloc(&d_Ix,  total_nopad*sizeof(float));
    cudaMalloc(&d_Iy,  total_nopad*sizeof(float));
    cudaMalloc(&d_Ixx, total_nopad*sizeof(float));
    cudaMalloc(&d_Iyy, total_nopad*sizeof(float));
    cudaMalloc(&d_Ixy, total_nopad*sizeof(float));

    double* d_outA; // A = 2x2 = 4 doubles per pixel
    double* d_outB; // B = 2 doubles per pixel
    double* d_outC; // c = 1 double per pixel
    cudaMalloc(&d_outA, total_nopad * 4 * sizeof(double));
    cudaMalloc(&d_outB, total_nopad * 2 * sizeof(double));
    cudaMalloc(&d_outC, total_nopad * sizeof(double));   
    // stride = 1 in polyExpansion, no downsampling
    int filterSize = 2*polyN + 1;      // 11
    constexpr int stride = 1;
    dim3 threads(TILE_COLS, TILE_ROWS);
    dim3 blocks((width - 2*polyN + TILE_COLS - 1) / TILE_COLS, 
                (height - 2*polyN + TILE_ROWS - 1) / TILE_ROWS);
    // for now we dont define shared memory  as it's allocated inside the Conv1D kernel
    // Launch on 6 parallel streams to maximize the parallelization of 6 parallel convolution
    convolution1DKernel<stride,HALF_POLY_KERNEL_SIZE><<<blocks,threads,0,streams[0]>>>(d_in, 0, 0, d_C, height, width, filterSize);         //cg, cg
    convolution1DKernel<stride,HALF_POLY_KERNEL_SIZE><<<blocks,threads,0,streams[1]>>>(d_in, 0, 1, d_Iy, height, width, filterSize);   //c_g, c_xg
    convolution1DKernel<stride,HALF_POLY_KERNEL_SIZE><<<blocks,threads,0,streams[2]>>>(d_in, 1, 0, d_Ix, height, width, filterSize);   //c_xg, c_g
    convolution1DKernel<stride,HALF_POLY_KERNEL_SIZE><<<blocks,threads,0,streams[3]>>>(d_in, 0, 2, d_Iyy, height, width, filterSize); //c_g, c_xxg
    convolution1DKernel<stride,HALF_POLY_KERNEL_SIZE><<<blocks,threads,0,streams[4]>>>(d_in, 2, 0, d_Ixx, height, width, filterSize); //c_xxg, c_g
    convolution1DKernel<stride,HALF_POLY_KERNEL_SIZE><<<blocks,threads,0,streams[5]>>>(d_in, 1, 1, d_Ixy, height, width, filterSize); //c_xg, c_xg
    // wait for all streams to syncronize
    for(int i = 0; i < 6; ++i)
        cudaStreamSynchronize(streams[i]);

    // clean up the streams
    for(int i = 0; i < 6; ++i)
        cudaStreamDestroy(streams[i]);

    // Caluclate matrix A,B,C on GPU for eahc pixel ( 1 thread per pixel)
    computePolyCoefficients<<<blocks, threads>>>(d_Ix, d_Iy, d_Ixx, d_Iyy, d_Ixy, d_C, d_outA, d_outB, d_outC, width-2*polyN, height-2*polyN); //, d_ig11, d_ig03, d_ig33, d_ig55);
    cudaDeviceSynchronize();
    cudaMemcpy(outA, d_outA, total_nopad * 4 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(outB, d_outB, total_nopad * 2 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(outC, d_outC, total_nopad * sizeof(double),   cudaMemcpyDeviceToHost);

    cudaMemcpy(outdc,  d_C,   total_nopad*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outIx,  d_Ix,  total_nopad*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outIy,  d_Iy,  total_nopad*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outIxx, d_Ixx, total_nopad*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outIyy, d_Iyy, total_nopad*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outIxy, d_Ixy, total_nopad*sizeof(float), cudaMemcpyDeviceToHost);

    // free the temp output malloced memoreis
    cudaFree(d_C);   
    cudaFree(d_Ix); 
    cudaFree(d_Iy);
    cudaFree(d_Ixx); 
    cudaFree(d_Iyy); 
    cudaFree(d_Ixy);
    cudaFree(d_outA);
    cudaFree(d_outB);
    cudaFree(d_outC);
    cudaFree(d_in); 
 }
