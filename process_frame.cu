#include <cuda_runtime.h>
#include <stdio.h>

__global__ void invert_colors(float* input, float* output, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = height * width;
    if (idx < total) {
        output[idx] = 1 - input[idx];  // Simple invert
    }
}

extern "C"
void process_frame(float* input, float* output, int height, int width) {
    int total = height * width;
    float* d_in;
    float* d_out;
    printf("height=%d\n", height);
    printf("width=%d\n", width);
    printf("total=%d\n", total);

    cudaMalloc(&d_in, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));

    cudaMemcpy(d_in, input, total*sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    invert_colors<<<blocks, threads>>>(d_in, d_out, height, width);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, total*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

