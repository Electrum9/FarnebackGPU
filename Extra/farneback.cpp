#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
    // Path to video file (you can also pass this as a command-line argument)
    std::string videoPath = "file:///afs/ece.cmu.edu/usr/rtafresh/Private/FarnebackGPU/walking.mp4";
    if(argc > 1) {
        videoPath = argv[1];
    }

    // Desired resize dimensions (set to 0 to keep original size)
    int resizeWidth = 320;
    int resizeHeight = 180;

    // Open the video file
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video: " << videoPath << endl;
        return -1;
    }

    // Read the first frame
    Mat prevFrame;
    if (!cap.read(prevFrame)) {
        cerr << "Error: Could not read the first frame." << endl;
        return -1;
    }

    // Resize if needed
    if (resizeWidth > 0 && resizeHeight > 0) {
        resize(prevFrame, prevFrame, Size(resizeWidth, resizeHeight));
    }

    // Convert to grayscale
    Mat prevGray;
    cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

    // Get video FPS (default to 30 if not available)
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0)
        fps = 30;

    // Prepare output video writer
    string outputFilename = "optical_flow_output.mp4";
    int fourcc = VideoWriter::fourcc('m','p','4','v');
    Size frameSize(prevGray.cols, prevGray.rows);
    VideoWriter writer(outputFilename, fourcc, fps, frameSize);
    if (!writer.isOpened()) {
        cerr << "Error: Could not open output video for write." << endl;
        return -1;
    }

    // Create container for HSV visualization
    // We'll build a 3-channel 8-bit HSV image where:
    //   - Hue: derived from the angle (scaled appropriately)
    //   - Saturation: set to max (255)
    //   - Value: normalized magnitude of the flow
    Mat hsvImage(prevFrame.size(), CV_8UC3);
    vector<Mat> hsvChannels(3);
    // Prepare saturation channel (always 255)
    hsvChannels[1] = Mat::ones(prevGray.size(), CV_8U) * 255;

    // Variables for timing
    long long totalMicroseconds = 0;
    int frameCount = 0;

    cout << "Processing video with Farneback optical flow..." << endl;

    while (true) {
        Mat frame;
        if (!cap.read(frame))
            break; // End of video

        // Resize frame if needed
        if (resizeWidth > 0 && resizeHeight > 0) {
            resize(frame, frame, Size(resizeWidth, resizeHeight));
        }

        // Convert current frame to grayscale
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Start timing the optical flow computation
        auto start = chrono::steady_clock::now();

        // Compute optical flow using Farneback method
        // Parameters: pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        Mat flow;
        calcOpticalFlowFarneback(prevGray, gray, flow,
                                 0.5, 3, 15, 3, 5, 1.2, 0);

        auto end = chrono::steady_clock::now();
        double elapsedSeconds = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e6;
        totalMicroseconds += chrono::duration_cast<chrono::microseconds>(end - start).count();
        frameCount++;

        // Split flow into x and y components and convert to polar coordinates
        Mat flowParts[2];
        split(flow, flowParts);
        Mat magnitude, angle;
        // Use radians here (false) so we can convert to degrees manually
        cartToPolar(flowParts[0], flowParts[1], magnitude, angle, false);

        // Convert angle from radians to degrees and scale (divide by 2 as in the Python example)
        Mat hue;
        hue = angle * (180.0 / CV_PI / 2.0);
        hue.convertTo(hue, CV_8U);

        // Normalize the magnitude to the range 0 to 255 and convert to 8-bit
        Mat normMag;
        normalize(magnitude, normMag, 0, 255, NORM_MINMAX);
        normMag.convertTo(normMag, CV_8U);

        // Assemble HSV image: H from hue, S already set, V from normMag
        hsvChannels[0] = hue;
        hsvChannels[2] = normMag;
        merge(hsvChannels, hsvImage);

        // Convert HSV to BGR for visualization
        Mat bgrFlow;
        cvtColor(hsvImage, bgrFlow, COLOR_HSV2BGR);

        // Write the result to the output video
        writer.write(bgrFlow);

        // For display (optional)
        imshow("Optical Flow", bgrFlow);
        if (waitKey(1) == 27) // Exit if 'Esc' is pressed
            break;

        // Set the current gray frame as the previous one for next iteration
        prevGray = gray.clone();
    }

    cap.release();
    writer.release();

    double totalSeconds = totalMicroseconds / 1e6;
    double avgTimePerFrame = totalSeconds / frameCount;
    cout << "Processed " << frameCount << " frames." << endl;
    cout << "Total Optical Flow Computation Time: " << totalSeconds << " sec" << endl;
    cout << "Average Time per Frame: " << avgTimePerFrame << " sec" << endl;
    cout << "Optical flow video saved as: " << outputFilename << endl;

    return 0;
}
