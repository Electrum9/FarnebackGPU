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

#define TILE_ROWS 32 // 32 threads
#define TILE_COLS 32 // 32 threads
__constant__ float gaussianBlur_filter[5] = {1.0f/16, 4.0f/16, 6.0f/16, 4.0f/16, 1.0f/16};

__constant__ float c_ig11;
__constant__ float c_ig03;
__constant__ float c_ig33;
__constant__ float c_ig55;
__constant__ float c_g[11];
__constant__ float c_xg[11];
__constant__ float c_xxg[11];


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

    int NUM_BANKS = 32;

    int outx = bx * bdx + tx; // x-coordinate of the output
    int outy = by * bdy + ty; // y-coordinate of the output
    // if (tx == 0 && ty == 0 && bx==0 && by==0) {
    //     printf("Device sees filter taps: ");
    //     for(int k = -ksizeHalf; k <= ksizeHalf; ++k) {
    //         float tapv = ((filter_size==5)
    //                      ? gaussianBlur_filter[k+ksizeHalf]
    //                      : vFilter[k+ksizeHalf]);
    //         float taph = ((filter_size==5)
    //         ? gaussianBlur_filter[k+ksizeHalf]
    //         : hFilter[k+ksizeHalf]);
    //         printf("%f ....%f", tapv, taph);
    //     }
    //     printf("\n");
    // }
    // if (bx==0 && by==0 && tx==0 && ty==0) {
    //     printf("Dev c_g: ");
    //     for(int k = -ksizeHalf; k<=ksizeHalf; ++k) {
    //         printf("%0.4f ", c_g[k+ksizeHalf]);
    //     }
    //     printf("\\n");
    // }
    __syncthreads();
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

    output[outy * (input_width - 2*ksizeHalf) + outx] = result; // use oriignal width without padding!!!!!
}


static void invert6(const double G_in[6][6], double invG_out[6][6]) {
    double M[6][12];
    for(int i=0;i<6;i++){
        for(int j=0;j<6;j++){
            M[i][j]   = G_in[i][j];
            M[i][j+6] = (i==j ? 1.0 : 0.0);
        }
    }
    for(int k=0;k<6;k++){
        double piv = M[k][k];
        // assume non-singular
        for(int j=0;j<12;j++) M[k][j] /= piv;
        for(int i=0;i<6;i++){
            if(i==k) continue;
            double f = M[i][k];
            for(int j=0;j<12;j++) M[i][j] -= f * M[k][j];
        }
    }
    for(int i=0;i<6;i++) for(int j=0;j<6;j++) invG_out[i][j] = M[i][j+6];
}

// this is on CPU host beforee launching the kernels
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
    // build the 6×6 moment matrix G and invert it, extract ig11, ig03, ig33, ig55…
    //Mat_<double> G(6, 6);
    //G.setTo(0);
    double G[6][6] = {{0}};

    for (int y = -n; y <= n; y++)
    {
        for (int x = -n; x <= n; x++)
        {
            G[0][0] += g[y]*g[x];
            G[1][1] += g[y]*g[x]*x*x;
            G[3][3] += g[y]*g[x]*x*x*x*x;
            G[5][5] += g[y]*g[x]*x*x*y*y;
        }
    }

    // //G[0][0] = 1.;
    // G(2,2) = G(0,3) = G(0,4) = G(3,0) = G(4,0) = G(1,1);
    // G(4,4) = G(3,3);
    // G(3,4) = G(4,3) = G(5,5);

    // // invG:
    // // [ x        e  e    ]
    // // [    y             ]
    // // [       y          ]
    // // [ e        z       ]
    // // [ e           z    ]
    // // [                u ]
    // Mat_<double> invG = G.inv(DECOMP_CHOLESKY);

    // ig11 = invG(1,1);
    // ig03 = invG(0,3);
    // ig33 = invG(3,3);
    // ig55 = invG(5,5);
    // G[2][2] = G[1][1];
    // G[0][3] = G[0][4] = G[3][0] = G[4][0] = G[1][1];
    // G[4][4] = G[3][3];
    // G[3][4] = G[4][3] = G[5][5];
    // double invG[6][6];
    // invert6(G, invG);
    // ig11 = invG[1][1];
    // ig03 = invG[0][3];
    // ig33 = invG[3][3];
    // ig55 = invG[5][5];
    ig03 = 0;
    ig11 = 0;
    ig33 = 0;
    for(int i=-n; i<=n; ++i){
    double gi = g[i+n];
    printf("gi is %f ", gi);
    ig03  +=   gi*   gi;        // ∑ g(x)^2
    ig11  += (i*gi)*(i*gi);     // ∑ [x·g(x)]^2  = ∑ x²·g(x)²
    ig33  += (i*i*gi)*(i*i*gi); // ∑ [x²·g(x)]^2 = ∑ x⁴·g(x)²
    }
    printf("\n");
    ig55 = ig11;
    printf("ig03: %f, ig11: %f, ig33: %f, ig55: %f\n", ig03, ig11, ig33, ig55);
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
 __host__ void setPolynomialExpansionConsts( int polyN, const float *g, const float *xg, const float *xxg, float ig11, float ig03, float ig33, float ig55)
{
    cudaSafeCall(cudaMemcpyToSymbol(c_g, g, (2*polyN + 1) * sizeof(*g)));
    cudaSafeCall(cudaMemcpyToSymbol(c_xg, xg, (2*polyN + 1) * sizeof(*xg)));
    cudaSafeCall(cudaMemcpyToSymbol(c_xxg, xxg, (2*polyN + 1) * sizeof(*xxg)));
    cudaSafeCall(cudaMemcpyToSymbol(c_ig11, &ig11, sizeof(ig11)));
    cudaSafeCall(cudaMemcpyToSymbol(c_ig03, &ig03, sizeof(ig03)));
    cudaSafeCall(cudaMemcpyToSymbol(c_ig33, &ig33, sizeof(ig33)));
    cudaSafeCall(cudaMemcpyToSymbol(c_ig55, &ig55, sizeof(ig55)));
}


__global__ void computePolyCoefficients( float* d_Ix, float* d_Iy, float* d_Ixx, float* d_Iyy, float* d_Ixy, float* d_C, float* d_A, float* d_B, float* d_c, int width, int height, float ig11, float ig03, float ig33, float ig55)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    // A matrix (2x2)
    float Axy =  0.5 * ig55 * d_Ixy[idx];
    d_A[4 * idx + 0] = ig03 * d_C[idx] + ig33 * d_Ixx[idx]; // +ig55 * d_Iyy[idx]; //Axx
    d_A[4 * idx + 1] = Axy; //Axy = Ayx
    d_A[4 * idx + 2] = Axy; //Ayx = Axy
    d_A[4 * idx + 3] = ig03 * d_C[idx] + ig33 * d_Iyy[idx];// + ig55 * d_Ixx[idx]; //Ayy

    // B vector
    d_B[2 * idx + 0] = d_Ix[idx] * ig11;  //Bx
    d_B[2 * idx + 1] = d_Iy[idx] * ig11;  //By

    // Scalar c
    d_c[idx] = d_C[idx] * ig03;  //smoothed Gaussian image 
}

extern "C"
void polynomialExpansion(const float* d_src, float sigma, int width, int height, int polyN, float* outC, float* outB,float* outA, float* outdc, float* outIx, float* outIy, float* outIxx, float* outIyy, float* outIxy)
 // will call convolution1D kernel it's from host CPU, launches 6 stream to run kernels for conv1D
 {  
    // Get the Polynomial Expansion Basis Gaussian filters 
    int total = height * width;
    int total_nopad = (height - 2*polyN) * (width - 2*polyN);
    std::vector<float> g, xg, xxg;
    double ig11, ig03, ig33, ig55;
    prepareGaussian(polyN, sigma, g, xg, xxg, ig11, ig03, ig33, ig55);
    setPolynomialExpansionConsts(polyN, g.data(), xg.data(), xxg.data(), float(ig11), float(ig03), float(ig33), float(ig55));
    float d_ig11 = float(ig11);
    float d_ig03 = float(ig03);
    float d_ig33 = float(ig33);
    float d_ig55 = float(ig55);
    float *d_g, *d_xg, *d_xxg;
    cudaMalloc(&d_g,  (2*polyN+1)*sizeof(float));
    cudaMalloc(&d_xg, (2*polyN+1)*sizeof(float));
    cudaMalloc(&d_xxg,(2*polyN+1)*sizeof(float));
    cudaMemcpy(d_g,  g.data(),  (2*polyN+1)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xg, xg.data(), (2*polyN+1)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xxg,xxg.data(),(2*polyN+1)*sizeof(float), cudaMemcpyHostToDevice);
    // Create 6 streams for each convolution calls 
    cudaStream_t streams[6];
    for(int i = 0; i < 6; ++i)
        cudaStreamCreate(&streams[i]);
    float* d_in;
    cudaMalloc(&d_in, total * sizeof(float));

    // 2) allocate six device buffers for the 6 raw responses
    // r1, r2, .....r6
    float *d_C, *d_Ix, *d_Iy, *d_Ixx, *d_Iyy, *d_Ixy;
    cudaMalloc(&d_C,   total_nopad*sizeof(float));
    cudaMalloc(&d_Ix,  total_nopad*sizeof(float));
    cudaMalloc(&d_Iy,  total_nopad*sizeof(float));
    cudaMalloc(&d_Ixx, total_nopad*sizeof(float));
    cudaMalloc(&d_Iyy, total_nopad*sizeof(float));
    cudaMalloc(&d_Ixy, total_nopad*sizeof(float));

    float* d_outA; // A = 2x2 = 4 floats
    float* d_outB; // B = 2 floats
    float* d_outC; // c = 1 float
    cudaMalloc(&d_outA, total_nopad * 4 * sizeof(float));
    cudaMalloc(&d_outB, total_nopad * 2 * sizeof(float));
    cudaMalloc(&d_outC, total_nopad * sizeof(float));   

    cudaMemcpy(d_in, d_src, total*sizeof(float), cudaMemcpyHostToDevice);

    #define POLY_KERNEL_SIZE 11
    #define HALF_POLY_KERNEL_SIZE 5
    int filterSize = 2*polyN + 1;      // 11
    int halfK      = polyN;            // 5
    constexpr int stride = 1;
    dim3 threads(TILE_COLS, TILE_ROWS);
    dim3 blocks((width - 2*polyN + TILE_COLS - 1) / TILE_COLS, 
                (height - 2*polyN + TILE_ROWS - 1) / TILE_ROWS);
    // for now we dont define shared memory  as it's allocated inside the Conv1D kernel
    convolution1DKernel<stride,HALF_POLY_KERNEL_SIZE><<<blocks,threads,0,streams[0]>>>(d_in, d_g, d_g, d_C, height, width, filterSize);
    convolution1DKernel<stride,HALF_POLY_KERNEL_SIZE><<<blocks,threads,0,streams[1]>>>(d_in, d_g, d_xg, d_Ix, height, width, filterSize);
    convolution1DKernel<stride,HALF_POLY_KERNEL_SIZE><<<blocks,threads,0,streams[2]>>>(d_in, d_xg, d_g, d_Iy, height, width, filterSize);
    convolution1DKernel<stride,HALF_POLY_KERNEL_SIZE><<<blocks,threads,0,streams[3]>>>(d_in, d_g, d_xxg, d_Ixx, height, width, filterSize);
    convolution1DKernel<stride,HALF_POLY_KERNEL_SIZE><<<blocks,threads,0,streams[4]>>>(d_in, d_xxg, d_g, d_Iyy, height, width, filterSize);
    convolution1DKernel<stride,HALF_POLY_KERNEL_SIZE><<<blocks,threads,0,streams[5]>>>(d_in, d_xg, d_xg, d_Ixy, height, width, filterSize);
    // wait for all streams
    for(int i = 0; i < 6; ++i)
        cudaStreamSynchronize(streams[i]);

    // clean up
    for(int i = 0; i < 6; ++i)
        cudaStreamDestroy(streams[i]);

        
    computePolyCoefficients<<<blocks, threads>>>(d_Ix, d_Iy, d_Ixx, d_Iyy, d_Ixy, d_C, d_outA, d_outB, d_outC, width-2*polyN, height-2*polyN, d_ig11, d_ig03, d_ig33, d_ig55);
    cudaDeviceSynchronize();
    cudaMemcpy(outA, d_outA, total_nopad * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outB, d_outB, total_nopad * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outC, d_outC, total_nopad * sizeof(float),   cudaMemcpyDeviceToHost);

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
    cudaFree(d_g);
    cudaFree(d_xg);
    cudaFree(d_xxg);
    cudaFree(d_C);
    cudaFree(d_Ix);
    cudaFree(d_Iy);
    cudaFree(d_Ixx);
    cudaFree(d_Iyy);
    cudaFree(d_Ixy);
 }


// extern "C"
// void process_frame(float* input, float* output, int height, int width) {
//     int total = height * width;
//     float* d_in;
//     float* d_out;
//     printf("height=%d\n", height);
//     printf("width=%d\n", width);
//     printf("total=%d\n", total);

//     cudaMalloc(&d_in, total * sizeof(float));
//     cudaMalloc(&d_out, total * sizeof(float));

//     cudaMemcpy(d_in, input, total*sizeof(float), cudaMemcpyHostToDevice);

//     // printf("First few input values:\n");
//     // for (int i = 0; i < 10 && i < total; ++i) {
//     //     printf("input[%d] = %f\n", i, input[i]);
//     // }

//     dim3 threads(TILE_COLS, TILE_ROWS);
//     dim3 blocks((width + TILE_COLS - 1) / TILE_COLS, 
//                 (height + TILE_ROWS - 1) / TILE_ROWS);
    
//     // Use the convolution kernel instead of invert_colors
//     int filter_size = 5; // Your filter is 5x5
//     //int halfK = filter_size/2;
//      // Using stride of 1 as in your code
//     #define GAUS_KERNEL_SIZE 5
//     #define POLY_KERNEL_SIZE 11
//     #define HALF_GAUS_KERNEL_SIZE 2
//     #define HALF_POLY_KERNEL_SIZE 5
//     convolution1DKernel<1,HALF_GAUS_KERNEL_SIZE><<<blocks, threads>>>(d_in, nullptr, nullptr, d_out, height, width, filter_size);
//     cudaDeviceSynchronize();
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("CUDA error: %s\n", cudaGetErrorString(err));
//     }

//     cudaMemcpy(output, d_out, total*sizeof(float), cudaMemcpyDeviceToHost);

//     // printf("First few output values:\n");
//     // for (int i = 0; i < 10 && i < total; ++i) {
//     //     printf("output[%d] = %f\n", i, output[i]);
//     // }

//     cudaFree(d_in);
//     cudaFree(d_out);
// }
