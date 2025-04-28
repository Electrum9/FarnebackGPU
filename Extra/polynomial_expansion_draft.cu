
#include <math.h>

#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y
#define bdx blockDim.x
#define bdy blockDim.y

#define TILE_ROWS 32 // 32 threads *
#define TILE_COLS 32 // 64 threads
__constant__ float gaussianBlur_filter[5] = {1.0f/16, 4.0f/16, 6.0f/16, 4.0f/16, 1.0f/16};
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

    output[outy * (input_width - 2*ksizeHalf) + outx] = result; // use oriignal width without padding!!!!!
}



#include <vector>
#include <cmath>
// this is on CPU host beforee launching the kernels
void prepareGaussian(int n, double sigma, std::vector<float>& g, std::vector<float>& xg, std::vector<float>& xxg,double& ig11, double& ig03, double& ig33, double& ig55)
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
    Mat_<double> G(6, 6);
    G.setTo(0);

    for (int y = -n; y <= n; y++)
    {
        for (int x = -n; x <= n; x++)
        {
            G(0,0) += g[y]*g[x];
            G(1,1) += g[y]*g[x]*x*x;
            G(3,3) += g[y]*g[x]*x*x*x*x;
            G(5,5) += g[y]*g[x]*x*x*y*y;
        }
    }

    //G[0][0] = 1.;
    G(2,2) = G(0,3) = G(0,4) = G(3,0) = G(4,0) = G(1,1);
    G(4,4) = G(3,3);
    G(3,4) = G(4,3) = G(5,5);

    // invG:
    // [ x        e  e    ]
    // [    y             ]
    // [       y          ]
    // [ e        z       ]
    // [ e           z    ]
    // [                u ]
    Mat_<double> invG = G.inv(DECOMP_CHOLESKY);

    ig11 = invG(1,1);
    ig03 = invG(0,3);
    ig33 = invG(3,3);
    ig55 = invG(5,5);
}

__constant__ float c_ig11;
__constant__ float c_ig03;
__constant__ float c_ig33;
__constant__ float c_ig55;
__constant__ float c_g[11];
__constant__ float c_xg[11];
__constant__ float c_xxg[11];


 // it's being called to copy the results of the preapreGaussian int constnat memories in the GPU
void setPolynomialExpansionConsts( int polyN, const float *g, const float *xg, const float *xxg, float ig11, float ig03, float ig33, float ig55)
{
    cudaSafeCall(cudaMemcpyToSymbol(c_g, g, (2*polyN + 1) * sizeof(*g)));
    cudaSafeCall(cudaMemcpyToSymbol(c_xg, xg, (2*polyN + 1) * sizeof(*xg)));
    cudaSafeCall(cudaMemcpyToSymbol(c_xxg, xxg, (2*polyN + 1) * sizeof(*xxg)));
    cudaSafeCall(cudaMemcpyToSymbol(c_ig11, &ig11, sizeof(ig11)));
    cudaSafeCall(cudaMemcpyToSymbol(c_ig03, &ig03, sizeof(ig03)));
    cudaSafeCall(cudaMemcpyToSymbol(c_ig33, &ig33, sizeof(ig33)));
    cudaSafeCall(cudaMemcpyToSymbol(c_ig55, &ig55, sizeof(ig55)));
}


void polynomialExpansion(const float* d_src, int width, int height, int polyN, float* outC, float* outB,float* outA)
 // will call convolution1D kernel it's from host CPU, launches 6 stream to run kernels for conv1D
 {  
    // Get the Polynomial Expansion Basis Gaussian filters 
    int total = height * width;
    std::vector<float> g, xg, xxg;
    double ig11, ig03, ig33, ig55;
    prepareGaussian(polyN, sigma, g, xg, xxg, ig11, ig03, ig33, ig55);
    setPolynomialExpansionConsts(polyN, g.data(), xg.data(), xxg.data(), float(ig11), float(ig03), float(ig33), float(ig55));
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
    cudaMalloc(&d_C,   total*sizeof(float));
    cudaMalloc(&d_Ix,  total*sizeof(float));
    cudaMalloc(&d_Iy,  total*sizeof(float));
    cudaMalloc(&d_Ixx, total*sizeof(float));
    cudaMalloc(&d_Iyy, total*sizeof(float));
    cudaMalloc(&d_Ixy, total*sizeof(float));

    float* d_outA; // A = 2x2 = 4 floats
    float* d_outB; // B = 2 floats
    float* d_outC; // c = 1 float
    cudaMalloc(&d_outA, total * 4 * sizeof(float));
    cudaMalloc(&d_outB, total * 2 * sizeof(float));
    cudaMalloc(&d_outC, total * sizeof(float))

    cudaMemcpy(d_in, input, total*sizeof(float), cudaMemcpyHostToDevice);

    #define POLY_KERNEL_SIZE 11
    #define HALF_POLY_KERNEL_SIZE 5
    int filter_size = 11
    int filterSize = 2*polyN + 1;      // 11
    int halfK      = polyN;            // 5
    constexpr int stride = 1;
    dim3 threads(TILE_COLS, TILE_ROWS);
    dim3 blocks((width + TILE_COLS - 1) / TILE_COLS, 
                (height + TILE_ROWS - 1) / TILE_ROWS);
    

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

        
    computePolyCoefficients<<<blocks, threads>>>(d_Ix, d_Iy, d_Ixx, d_Iyy, d_Ixy, d_C, d_outA, d_outB, d_outC, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(outA, d_outA, total * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outB, d_outB, total * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outC, d_outC, total * sizeof(float),   cudaMemcpyDeviceToHost);

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

 }


 __global__ void computePolyCoefficients( float* d_Ix, float* d_Iy, float* d_Ixx, float* d_Iyy, float* d_Ixy, float* d_C, float* d_A, float* d_B, float* d_c, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // A matrix (2x2)
    float Axy = c_ig55 * d_Ixy[idx];
    d_A[4 * idx + 0] = c_ig03 * d_C + c_ig33 * d_Ixx[idx]; //Axx
    d_A[4 * idx + 1] = Axy; //Axy = Ayx
    d_A[4 * idx + 2] = Axy; //Ayx = Axy
    d_A[4 * idx + 3] = c_ig03 * d_C + c_ig33 * d_Iyy[idx]; //Ayy

    // B vector
    d_B[2 * idx + 0] = d_Ix[idx] * c_ig11;  //Bx
    d_B[2 * idx + 1] = d_Iy[idx] * c_ig11;  //By

    // Scalar c
    d_c[idx] = d_C[idx];  //smoothed Gaussian image 
}

     
    // dst(y, xWarped) = b3*c_ig11;   //By
    // dst(height + y, xWarped) = b2*c_ig11;  //Bx
    // dst(2*height + y, xWarped) = b1*c_ig03 + b5*c_ig33; //xx
    // dst(3*height + y, xWarped) = b1*c_ig03 + b4*c_ig33; //Ayy
    // dst(4*height + y, xWarped) = b6*c_ig55; //Axy = Ayx
    
    // const float* R0, // d_C   (I * g * g') b1
    // const float* R1, // d_Ix  (I * xg * g') b2
    // const float* R2, // d_Iy  (I * g * xg') b3
    // const float* R3, // d_Ixx (I * xxg * g') b4 
    // const float* R4, // d_Iyy (I * g * xxg') b5
    // const float* R5, // d_Ixy (I * xg * xg') b6



