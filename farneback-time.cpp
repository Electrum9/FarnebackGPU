#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <iostream>
#include <chrono>

int main() {
    // Open video file
    cv::VideoCapture cap(cv::samples::findFile("vtest.avi"));
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video.\n";
        return -1;
    }

    cv::Mat frame1, frame2, gray1, gray2;
    cap >> frame1;
    if (frame1.empty()) {
        std::cerr << "Error: Could not read first frame.\n";
        return -1;
    }

    // Convert to grayscale
    cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);

    // CUDA GPU Mats
    cv::cuda::GpuMat d_gray1, d_gray2, d_flow;

    // CUDA Farneback Optical Flow
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> opticalFlow = cv::cuda::FarnebackOpticalFlow::create();

    float totalTime = 0.0f;
    int frameCount = 0;

    while (true) {
        cap >> frame2;
        if (frame2.empty()) break;

        cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

        // Upload to GPU
        d_gray1.upload(gray1);
        d_gray2.upload(gray2);

        // CUDA Events for Timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        opticalFlow->calc(d_gray1, d_gray2, d_flow);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;
        frameCount++;

        std::cout << "Frame " << frameCount << " took " << milliseconds << " ms\n";

        // Cleanup events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Swap buffers
        std::swap(gray1, gray2);
    }

    std::cout << "Average Optical Flow Time: " << (totalTime / frameCount) << " ms per frame\n";
    return 0;
}
