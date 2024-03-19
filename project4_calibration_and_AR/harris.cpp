/**
 * author: Harshit Kumar, Khushi Neema
 * date: Mar 5th, 2024
 * purpose: Detects Harris corners in a video stream
 *
 */

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using std::vector;

int main() {
    // Open the default video camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }

    namedWindow("Harris Corners", WINDOW_AUTOSIZE);

    // Parameters for Harris corner detection
    int blockSize = 5; // Size of the neighborhood considered for corner detection
    int apertureSize = 3; // Aperture parameter for the Sobel operator.
    double k = 0.04; // Harris detector free parameter
    int thresh = 100; // Threshold for detecting corners

    Mat frame;
    while (true) {
        cap >> frame; // Capture frame-by-frame
        if (frame.empty())
            break;

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // Detecting Harris corners
        Mat dst = Mat::zeros(frame.size(), CV_32FC1);
        cornerHarris(gray, dst, blockSize, apertureSize, k);

        // Normalizing & converting to a format that can be visualized
        Mat dst_norm, dst_norm_scaled;
        normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
        convertScaleAbs(dst_norm, dst_norm_scaled);

        // Drawing a circle around corners
        for (int i = 0; i < dst_norm.rows; i++) {
            for (int j = 0; j < dst_norm.cols; j++) {
                if ((int)dst_norm.at<float>(i, j) > thresh) {
                    circle(frame, Point(j, i), 5, Scalar(255, 0, 0), 2, 8, 0);
                }
            }
        }

        // Displaying the result
        imshow("Harris Corners", frame);

        if (waitKey(30) >= 0) break; // Wait for a keystroke in the window
    }

    // When everything done, release the video capture object
    cap.release();
    destroyAllWindows();
    return 0;
}
