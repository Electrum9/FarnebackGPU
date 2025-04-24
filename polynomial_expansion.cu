
#include <math.h>

#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y
#define bdx blockDim.x
#define bdy blockDim.y




// Q: Will filter be hardcoded (aka filter size) --> do they all require the same size? 
    // have it stored in an array in global memory (5x5, 3x3) 
    // https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html
    // each thread will compute the part of the filter it needs (which row/which column)
    // ** just define the filter in the kernel so each one has it in register memory --> the thread can figure out which row it means
        // Pros: avoid computation time, kind of like a look up 
        // Cons: have to store in global memory -> shared memory (potential bank conflict) vs each thread computing it themselves
        // Pros: avoid computation time, kind of like a look up 
        // Cons: have to store in global memory -> shared memory (potential bank conflict) vs each thread computing it themselves
// stride is variable? Yes --> should be 1 or 2
// assumption: padding will be done on the CPU before passing into the GPU kernel --> have boundary checks in the logic
    // warp divergence with different branches? 

    //** for now, STRIDE = 1 */
// __global__ void convolution1DKernel(float* input, float* filter, float* output, 
//                                     int input_height, int input_width,
//                                     int filter_size, int stride)



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

void setPolynomialExpansionConsts(    // it's being called to copy the results of the preapreGaussian int constnat memories in the GPU
    int polyN, const float *g, const float *xg, const float *xxg,
    float ig11, float ig03, float ig33, float ig55)
{
cudaSafeCall(cudaMemcpyToSymbol(c_g, g, (2*polyN + 1) * sizeof(*g)));
cudaSafeCall(cudaMemcpyToSymbol(c_xg, xg, (2*polyN + 1) * sizeof(*xg)));
cudaSafeCall(cudaMemcpyToSymbol(c_xxg, xxg, (2*polyN + 1) * sizeof(*xxg)));
cudaSafeCall(cudaMemcpyToSymbol(c_ig11, &ig11, sizeof(ig11)));
cudaSafeCall(cudaMemcpyToSymbol(c_ig03, &ig03, sizeof(ig03)));
cudaSafeCall(cudaMemcpyToSymbol(c_ig33, &ig33, sizeof(ig33)));
cudaSafeCall(cudaMemcpyToSymbol(c_ig55, &ig55, sizeof(ig55)));
}






void polynomialexpansion()   
 // will call convolution1D kernel it's from host CPU, launches 6 stream to run kernels for conv1D
 {  

    int stride = 1;
    int filter_size = 11
    dim3 threads(TILE_COLS, TILE_ROWS);
    dim3 blocks((width + TILE_COLS - 1) / TILE_COLS, 
                (height + TILE_ROWS - 1) / TILE_ROWS);
    

    // for now we dont define shared memory  as it's allocated inside the Conv1D kernel




 }

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
 