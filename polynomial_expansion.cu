


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
__global__ void convolution1DKernel(float* input, float* filter, float* output, 
                                    int input_height, int input_width,
                                    int filter_size, int stride)
{
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


// how many thread blocks, how many threads per thread block? how many SMs?
// come back to this 
convolution1DKernel<<<1, 1>>>(dev_out);