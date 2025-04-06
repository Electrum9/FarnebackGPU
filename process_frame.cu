#include <cuda_runtime.h>
#include <stdio.h>

__global__ void invert_colors(unsigned char* input, unsigned char* output, int height, int width, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = height * width * channels;
    if (idx < total) {
        output[idx] = 255 - input[idx];  // Simple invert
    }
}

extern "C"
void process_frame(unsigned char* input, unsigned char* output, int height, int width, int channels) {
    int total = height * width * channels;
    unsigned char* d_in;
    unsigned char* d_out;

    cudaMalloc((void**)&d_in, total);
    cudaMalloc((void**)&d_out, total);

    cudaMemcpy(d_in, input, total, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    invert_colors<<<blocks, threads>>>(d_in, d_out, height, width, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, total, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

