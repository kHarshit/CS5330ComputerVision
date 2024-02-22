#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "objDetect.h"

using namespace cv;

class UnionFind {
private:
    std::vector<int> parent;
    std::vector<int> rank;

public:
    UnionFind(int n) {
        parent.resize(n);
        rank.resize(n);
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
            rank[i] = 0;
        }
    }

    int find(int u) {
        if (parent[u] != u) {
            parent[u] = find(parent[u]); // Path compression
        }
        return parent[u];
    }

    void unite(int u, int v) {
        int rootU = find(u);
        int rootV = find(v);
        if (rootU == rootV) return;

        if (rank[rootU] < rank[rootV]) {
            parent[rootU] = rootV;
        } else if (rank[rootU] > rank[rootV]) {
            parent[rootV] = rootU;
        } else {
            parent[rootV] = rootU;
            rank[rootU]++;
        }
    }
};

void connectedComponents2(const Mat& binaryImage, Mat& labeledImage) {
    labeledImage = Mat::zeros(binaryImage.size(), CV_32S); // Initialize labeled image

    int rows = binaryImage.rows;
    int cols = binaryImage.cols;
    UnionFind uf(rows * cols);

    int label = 1; // Start labeling from 1

    // Traverse the image pixels
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (binaryImage.at<uchar>(i, j) != 0) {
                int current = i * cols + j;
                int up = (i > 0) ? (current - cols) : -1;
                int left = (j > 0) ? (current - 1) : -1;

                // Union with neighboring pixels
                if (up != -1 && binaryImage.at<uchar>(i - 1, j) != 0)
                    uf.unite(current, up);
                if (left != -1 && binaryImage.at<uchar>(i, j - 1) != 0)
                    uf.unite(current, left);
            }
        }
    }

    // Assign labels to connected components
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int current = i * cols + j;
            if (binaryImage.at<uchar>(i, j) != 0) {
                labeledImage.at<int>(i, j) = uf.find(current);
            }
        }
    }
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


        Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11));
        //Calling morphological filtering function
        cv::Mat cleaned;
        morphologyEx(thresholdedFrame,cleaned,MORPH_CLOSE,kernel);
        
        cv:Mat labeledImage(cleaned.size(), CV_32S); 
        cv::Mat labeledImage8U;
        
        connectedComponents2(cleaned,labeledImage);
        //cv::imshow("Defined Regions",labeledImage);
        labeledImage.convertTo(labeledImage8U, CV_8U);


        Mat coloredImage;
        applyColorMap(labeledImage8U, coloredImage, COLORMAP_JET);
        //std::cout<<labeledImage8U<<std::endl;

        cv::imshow("Original Video", frame);
        cv::imshow("Defined Regions", coloredImage);
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