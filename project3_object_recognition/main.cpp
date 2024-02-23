#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <map>
#include <vector>
#include <iostream>
#include <fstream>
#include "objDetect.h"

using namespace cv;
using namespace std;


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
    cv::namedWindow("Conected Components",WINDOW_NORMAL);
    cv::namedWindow("Connected Components Features",WINDOW_NORMAL);
    for (;;)
    {
        char key = cv::waitKey(10);

        // cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            std::cout << "Error: Blank frame grabbed" << std::endl;
            continue;
        }
        

        // 1. Preprocess and threshold the frame
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


        // 2. Clean the thresholded frame
        cv::Mat cleanedImg;
        // use 11x11 kernel
        // Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11));
        // use 3x3 8-connected kernel
        cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
        // or use 3x3 4-connected kernel
        // 0 1 0
        // 1 1 1
        // 0 1 0
        // cv::Mat kernel = cv::Mat::zeros(3, 3, CV_8U);
        // kernel.at<uchar>(1, 0) = 1;
        // kernel.at<uchar>(0, 1) = 1;
        // kernel.at<uchar>(1, 1) = 1;
        // kernel.at<uchar>(1, 2) = 1;
        // kernel.at<uchar>(2, 1) = 1;

        morphologyEx(thresholdedFrame,cleanedImg,MORPH_CLOSE,kernel);
        
        cv::Mat labeledImg(cleanedImg.size(), CV_32S); 
        cv::Mat labeledImgNormalized;
        
        // 3. Find connected components
        cv::Mat colorLabeledImg;
        // returns a map of connected components {pixel number: connected component number}
        std::map<int, int> connectedComponents = connectedComponentsTwoPass(cleanedImg, labeledImg);
        cv::normalize(labeledImg, labeledImgNormalized, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(labeledImgNormalized, colorLabeledImg, cv::COLORMAP_JET);

        // 4. Compute features for each connected component
        cv::Mat colorLabeledFeatureImg;
        cv::Mat featureOutImg;
        map<int, ObjectFeatures> featuresMap = computeFeatures(labeledImg, connectedComponents, featureOutImg);
        cout << "Number of connected components: " << featuresMap.size() << endl;
        cv::normalize(featureOutImg, featureOutImg, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(featureOutImg, colorLabeledFeatureImg, cv::COLORMAP_JET);

        // 5. Enable your system to collect feature vectors from objects, attach labels, and store them in an object DB (e.g. a file). In other words, your system needs to have a training mode that enables you to collect the feature vectors of known objects and store them for later use in classifying unknown objects. You may want to implement this as a response to a key press: when the user types an N, for example, have the system prompt the user for a name/label and then store the feature vector for the current object along with its label into a file. This could also be done from labeled still images of the object, such as those from a training set.
        if (key == 'n')
        {
            // store the feature vector for the current object along with its label into a file
            std::string filename = "object_db.txt";

            // iterate through the featuresMap and store the feature vectors
            // map<int, ObjectFeatures> featuresMap where ObjectFeatures is structstruct ObjectFeatures { double percentFilled; double aspectRatio;};
            for (const auto& featurePair : featuresMap) {
                std::cout << "Enter label for the object: ";
                std::string label;
                std::cin >> label;

                std::ofstream file(filename, std::ios::app); // Append mode
                file << label << "," << featurePair.second.percentFilled << "," << featurePair.second.aspectRatio << "\n";
                file.close();
            }
        }

        // Display the images
        cv::imshow("0. Original Video", frame);
        cv::imshow("1. Thresholded", thresholdedFrame);
        cv::imshow("2. Cleaned thresholded",cleanedImg);
        cv::imshow("3. Connected Components", colorLabeledImg);
        cv::imshow("4. Connected Components Features", colorLabeledFeatureImg);
        if (key == 'q')
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}