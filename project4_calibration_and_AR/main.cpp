#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "chessboardcorner.h"
using namespace cv;

int main()
{
     // Wait for a keystroke in the window
    cv::VideoCapture capdev(0);
    cv::Mat frame;
    cv::Size boardSize(9,6);
    cv::namedWindow("Display Window",WINDOW_NORMAL);
    
    if(!capdev.isOpened()){
        printf("Unable to open the video camera");
        return -1;
    }
    
    while(true)
    {
        capdev >> frame;

        if(frame.empty()){
            printf("Blank Frame grabbed");
            return -1;
        }
        Drawchessboardcorner(frame,boardSize);      //to find and display chessboard corners
        cv::imshow("Display Window",frame);

        char k = waitKey(10);
        if(k=='q'){                 //If the user presses 'q', end the program
        break;
        }
    }
    capdev.release();
    cv::destroyAllWindows();
    return 0;
}