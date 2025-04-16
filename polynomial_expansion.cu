
#include <math.h>

#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y
#define bdx blockDim.x
#define bdy blockDim.y

__global__ void polynomialexpansion(); // will call convolution1D kernel


// Q: Will filter be hardcoded (aka filter size) --> do they all require the same size? 
    // have it stored in an array in global memory (5x5, 3x3) 
    // https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html
    // each thread will compute the part of the filter it needs (which row/which column)
    // ** just define the filter in the kernel so each one has it in register memory --> the thread can figure out which row it means
        // Pros: avoid computation time, kind of like a look up 
        // Cons: have to store in global memory -> shared memory (potential bank conflict) vs each thread computing it themselves
// stride is variable? Yes --> should be 1 or 2
// assumption: padding will be done on the CPU before passing into the GPU kernel --> have boundary checks in the logic
    // warp divergence with different branches? 

    //** for now, STRIDE = 1 */
__global__ void convolution1DKernel(float* input, float* filter, float* output, 
                                    int input_height, int input_width,
                                    int filter_size, int stride)
{

    //how much shared memory to allocate --> shared memory for a thread block
    __shared__ float share_buf[];

    int numBanks = 32;
    int tileSizeH = 32;
    int tileSizeW = 64;

    // So we know where the start is in the output
    int numTiles = (int)ceilf(input_height/tileSizeH) * (int)ceilf(input_width/tileSizeW);
    int bigTileRow = bx/numTiles;
    int bigTileCol = by%numTiles;

    


    /*
    Pseudocodee:
        1. Figure out the index where the tile starts --> numTiles + offset
        2. We need to read in the data
        3. Do the horizontal pass 
        4. Synthreads
        5. Do the vertical pass
        6. Store back in output 
    
    */


    



    //horizontal pass --> store intermediate representation in global memory on GPU
    //syncthreads()
    //vertical pass
}

// reading data in from global to shared: each data is reading --> 
    // so one thread handles part of a row (blocking/tiling) --> need to make sure to not have bank conflicts
    // each thread block works on 32x32 patch (1024 threads to copy everything in parallel)
// Pros of using all threads to copy: fast but then quarter the threads are wasted not doing computation
// Cons of using quarter threads to copy: slow to copy but then no waste
// have as many threads as output --> each thread copies a patch

// when stride == 1 --> each pixel in the input contributes to an overlapping region



// how many thread blocks, how many threads per thread block? how many SMs?
// come back to this 
convolution1DKernel<<<1, 1>>>(dev_out);



////////////////OPENCV Version ////////////////////////////

#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y
#define bdx blockDim.x
#define bdy blockDim.y

#define BORDER_SIZE 5
#define MAX_KSIZE_HALF 100

__constant__ float c_gKer[MAX_KSIZE_HALF + 1];

    template <typename Border>
    __global__ void gaussianBlur(
            const int height, const int width, const PtrStepf src, const int ksizeHalf,
            const Border b, PtrStepf dst)
    {
        // Resolve to global coordinates in array
        const int y = by * bdy + ty;
        const int x = bx * bdx + tx;

        extern __shared__ float smem[];
        volatile float *row = smem + ty * (bdx + 2*ksizeHalf); // pointer to beginning of row
        // Get an entire row, including halo (which is 2*ksizeHalf)


        if (y < height)
        {
            // Vertical pass
            for (int i = tx; i < bdx + 2*ksizeHalf; i += bdx)
            {
                int xExt = int(bx * bdx) + i - ksizeHalf;
                xExt = b.idx_col(xExt); // Remap (?) xExt to place in image (depends on border type)
                row[i] = src(y, xExt) * c_gKer[0]; // Multiply central pixel by center kernel value (srx is image pizels in the global memory)
                for (int j = 1; j <= ksizeHalf; ++j)
                    row[i] +=
                            (src(b.idx_row_low(y - j), xExt) +
                             src(b.idx_row_high(y + j), xExt)) * c_gKer[j]; // Use symmetry of filter to cut multiplies in half
            }

            if (x < width)
            {
                __syncthreads();

                // Horizontal pass
                row += tx + ksizeHalf;
                float res = row[0] * c_gKer[0];
                for (int i = 1; i <= ksizeHalf; ++i)
                    res += (row[-i] + row[i]) * c_gKer[i];
                dst(y, x) = res;
            }
        }
    }


    void setGaussianBlurKernel(const float *gKer, int ksizeHalf)
    {
        cudaSafeCall(cudaMemcpyToSymbol(c_gKer, gKer, (ksizeHalf + 1) * sizeof(*gKer)));
    }


    template <typename Border>
    void gaussianBlurCaller(const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, cudaStream_t stream)
    {
        int height = src.rows;
        int width = src.cols;

        dim3 block(256);
        dim3 grid(divUp(width, block.x), divUp(height, block.y));
        int smem = (block.x + 2*ksizeHalf) * block.y * sizeof(float);
        Border b(height, width);

        gaussianBlur<<<grid, block, smem, stream>>>(height, width, src, ksizeHalf, b, dst);

        cudaSafeCall(cudaGetLastError());

        if (stream == 0)
            cudaSafeCall(cudaDeviceSynchronize());
    }


    void gaussianBlurGpu(
            const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, int borderMode, cudaStream_t stream)
    {
        typedef void (*caller_t)(const PtrStepSzf, int, PtrStepSzf, cudaStream_t);

        static const caller_t callers[] =
        {
            0 /*gaussianBlurCaller<BrdConstant<float> >*/,
            gaussianBlurCaller<BrdReplicate<float> >,
            0 /*gaussianBlurCaller<BrdReflect<float> >*/,
            0 /*gaussianBlurCaller<BrdWrap<float> >*/,
            gaussianBlurCaller<BrdReflect101<float> >
        };

        callers[borderMode](src, ksizeHalf, dst, stream);
    }


    template <typename Border>
    __global__ void gaussianBlur5(
            const int height, const int width, const PtrStepf src, const int ksizeHalf,
            const Border b, PtrStepf dst)
    {
        const int y = by * bdy + ty;
        const int x = bx * bdx + tx;

        extern __shared__ float smem[];

        const int smw = bdx + 2*ksizeHalf; // shared memory "width"
        volatile float *row = smem + 5 * ty * smw;

        if (y < height)
        {
            // Vertical pass
            for (int i = tx; i < bdx + 2*ksizeHalf; i += bdx)
            {
                int xExt = int(bx * bdx) + i - ksizeHalf;
                xExt = b.idx_col(xExt);

                #pragma unroll
                for (int k = 0; k < 5; ++k)
                    row[k*smw + i] = src(k*height + y, xExt) * c_gKer[0];

                for (int j = 1; j <= ksizeHalf; ++j)
                    #pragma unroll
                    for (int k = 0; k < 5; ++k)
                        row[k*smw + i] +=
                                (src(k*height + b.idx_row_low(y - j), xExt) +
                                 src(k*height + b.idx_row_high(y + j), xExt)) * c_gKer[j];
            }

            if (x < width)
            {
                __syncthreads();

                // Horizontal pass

                row += tx + ksizeHalf;
                float res[5];

                #pragma unroll
                for (int k = 0; k < 5; ++k)
                    res[k] = row[k*smw] * c_gKer[0];

                for (int i = 1; i <= ksizeHalf; ++i)
                    #pragma unroll
                    for (int k = 0; k < 5; ++k)
                        res[k] += (row[k*smw - i] + row[k*smw + i]) * c_gKer[i];

                #pragma unroll
                for (int k = 0; k < 5; ++k)
                    dst(k*height + y, x) = res[k];
            }
        }
    }


    template <typename Border, int blockDimX>
    void gaussianBlur5Caller(
            const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, cudaStream_t stream)
    {
        int height = src.rows / 5;
        int width = src.cols;

        dim3 block(blockDimX);
        dim3 grid(divUp(width, block.x), divUp(height, block.y));
        int smem = (block.x + 2*ksizeHalf) * 5 * block.y * sizeof(float);
        Border b(height, width);

        gaussianBlur5<<<grid, block, smem, stream>>>(height, width, src, ksizeHalf, b, dst);

        cudaSafeCall(cudaGetLastError());

        if (stream == 0)
            cudaSafeCall(cudaDeviceSynchronize());
    }


    void gaussianBlur5Gpu(
            const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, int borderMode, cudaStream_t stream)
    {
        typedef void (*caller_t)(const PtrStepSzf, int, PtrStepSzf, cudaStream_t);

        static const caller_t callers[] =
        {
            0 /*gaussianBlur5Caller<BrdConstant<float>,256>*/,
            gaussianBlur5Caller<BrdReplicate<float>,256>,
            0 /*gaussianBlur5Caller<BrdReflect<float>,256>*/,
            0 /*gaussianBlur5Caller<BrdWrap<float>,256>*/,
            gaussianBlur5Caller<BrdReflect101<float>,256>
        };

        callers[borderMode](src, ksizeHalf, dst, stream);
    }

    void gaussianBlur5Gpu_CC11(
            const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, int borderMode, cudaStream_t stream)
    {
        typedef void (*caller_t)(const PtrStepSzf, int, PtrStepSzf, cudaStream_t);

        static const caller_t callers[] =
        {
            0 /*gaussianBlur5Caller<BrdConstant<float>,128>*/,
            gaussianBlur5Caller<BrdReplicate<float>,128>,
            0 /*gaussianBlur5Caller<BrdReflect<float>,128>*/,
            0 /*gaussianBlur5Caller<BrdWrap<float>,128>*/,
            gaussianBlur5Caller<BrdReflect101<float>,128>
        };

        callers[borderMode](src, ksizeHalf, dst, stream);
    }

}}}} // namespace cv { namespace cuda { namespace cudev { namespace optflow_farneback


#endif /* CUDA_DISABLER */
