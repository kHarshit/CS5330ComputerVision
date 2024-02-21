#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "objDetect.h"

using namespace cv;
//2nd APPROACH
class UnionFind {
private:
    std::vector<int> parent;
public:
    UnionFind(int size) {
        parent.resize(size);
        for (int i = 0; i < size; ++i)
            parent[i] = i;
    }

    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }

    void merge(int x, int y) {
        parent[find(x)] = find(y);
    }
};

// Connected components analysis using two-pass algorithm with union-find
Mat connectedComponentsAnalysis(const Mat& binaryImage) {
    Mat labels(binaryImage.size(), CV_32S, Scalar(0));
    UnionFind uf(binaryImage.rows * binaryImage.cols);

    int label = 1; // Start labeling from 1
    std::unordered_map<int, int> labelMap; // Map to track merged labels

    // First Pass
    for (int i = 0; i < binaryImage.rows; ++i) {
        for (int j = 0; j < binaryImage.cols; ++j) {
            if (binaryImage.at<uchar>(i, j) > 0) { // Foreground pixel
                int upLabel = (i > 0) ? labels.at<int>(i - 1, j) : 0;
                int leftLabel = (j > 0) ? labels.at<int>(i, j - 1) : 0;

                if (upLabel == 0 && leftLabel == 0) { // New label
                    labels.at<int>(i, j) = label;
                    labelMap[label] = label;
                    label++;
                } else {
                    if (upLabel != 0 && leftLabel != 0 && upLabel != leftLabel) {
                        int rootUp = uf.find(upLabel);
                        int rootLeft = uf.find(leftLabel);
                        if (rootUp != rootLeft) { // Merge equivalence classes
                            uf.merge(rootUp, rootLeft);
                            labelMap[label] = min(rootUp, rootLeft); // Store merged label
                        }
                    }
                    int maxLabel = max(upLabel, leftLabel);
                    labels.at<int>(i, j) = maxLabel;
                    labelMap[maxLabel] = maxLabel; // Store label
                }
            }
        }
    }

    // Second Pass
    for (int i = 0; i < binaryImage.rows; ++i) {
        for (int j = 0; j < binaryImage.cols; ++j) {
            int currentLabel = labels.at<int>(i, j);
            if (currentLabel != 0) {
                labels.at<int>(i, j) = labelMap[uf.find(currentLabel)]; // Propagate correct label
            }
        }
    }

    return labels;
}

int main()
{
    // to open a default camera
    cv::VideoCapture cap(0);
    cv::Mat frame, blur, hsv;
    if (!cap.isOpened())
    {
        printf("Error opening the default camera");
        return -1;
    }
    cv::namedWindow("Original Video", WINDOW_NORMAL);
    cv::namedWindow("Thresholded", WINDOW_NORMAL);
    cv::namedWindow("Cleaned Threholded",WINDOW_NORMAL);
    cv::namedWindow("Defined Regions",WINDOW_NORMAL);
    for (;;)
    {

        // cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            std::cout << "Error: Blank frame grabbed" << std::endl;
            continue;
        }
        cv::imshow("Original Video", frame);

        // Preprocess and threshold the frame
        cv::Mat thresholdedFrame = preprocessAndThreshold(frame);

#if 0
        cv::imshow("Blurred Video",hsv);
        
        blur5x5_2(frame,blur);
        
        //cv::cvtColor(frame,frame,cv::COLOR_BGR2GRAY);
        cv::convertScaleAbs(blur, blur);
    
        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // // Dynamically set the threshold using ISODATA algorithm
        std::vector<Mat> channels;
        split(hsv, channels);
        Mat sat = channels[1]; // Saturation channel
        double minVal, maxVal;
        minMaxLoc(sat, &minVal, &maxVal);
        double thresholdValue = (minVal + maxVal) / 2.0;

        // // Thresholding
        Mat thresholdedFrame;
        cv::threshold(sat, thresholdedFrame, thresholdValue, 255, THRESH_BINARY);
#endif

        //To clean the noise/holes : we need external kernel, we have chosen a kernel filter of size 3X3
        // Mat kernel = (Mat_<int>(3,3) << 1, 1, 1,
        //                             1, 1, 1,
        //                             1, 1, 1);
        Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11));
        //Calling morphological filtering function
        cv::Mat cleaned;
        morphologyEx(thresholdedFrame,cleaned,MORPH_CLOSE,kernel);
        
        cv:Mat labeledImage(cleaned.size(), CV_32S); 
        cv::Mat labeledImage8U;
        labeledImage.convertTo(labeledImage8U, CV_8U);
        connectedComponents(cleaned,labeledImage);
        cv::imshow("Defined Regions",labeledImage);
        
        //2nd apprach
        labeledImage=connectedComponentsAnalysis(cleaned);
        cv::imshow("Defined Regions", labeledImage8U * (255 / labeledImage8U.rows));



        cv::imshow("Thresholded", thresholdedFrame);
        cv::imshow("Cleaned thresholded",cleaned);
        char key = cv::waitKey(10);
        if (key == 'q')
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}