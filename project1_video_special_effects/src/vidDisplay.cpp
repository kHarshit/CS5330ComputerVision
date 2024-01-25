/**
 * author: Harshit Kumar
 * date: Jan 18, 2024
 * purpose: Display video, allowing users to save frames and quit using keystrokes.
 * 
*/

#include <iostream>
#include "vidDisplay.h"
#include "filter.h"

using namespace std;

int displayVideo(int videoDeviceIndex) {
    // Open the video device
    cv::VideoCapture capdev(videoDeviceIndex);
    if (!capdev.isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }

    // Get some properties of the image
    cv::Size refS((int)capdev.get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // Identifies a window
    cv::Mat frame;
    cv::Mat filter = frame;

    char lastKeypress = '\0';  // Initialize the last keypress variable

    for (;;) {
        capdev >> frame; // Get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("Frame is empty\n");
            break;
        }

        // Check the last keypress and modify the image accordingly
        if (lastKeypress == 'g') {
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        } else if (lastKeypress == 'h') {
            // Use the custom greyscale function
            greyscale(frame, frame);
        } else if (lastKeypress == 't') {
            // Use the custom sepia function
            sepia(frame, frame);
        } else if (lastKeypress == 'b') {
            // Use the custom blur function
            blur5x5_1(frame, frame);
        } else if (lastKeypress == 'B') {
            // Use the custom (faster) blur function
            blur5x5_2(frame, filter);
            cv::convertScaleAbs(filter, frame);
        } else if (lastKeypress == 'x') {
            sobelX3x3(frame, filter);
            cv::convertScaleAbs(filter, frame);
            cv::imshow("SobelX", filter);
        } else if (lastKeypress == 'y') {
            sobelY3x3(frame, filter);
            cv::convertScaleAbs(filter, frame);
            cv::imshow("SobelY", filter);
        }

        cv::imshow("Video", frame);

        // check waiting keystroke
        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        } else if (key == 's') {
            cout << key << " pressed: Saving frame to captured_frame.png." << endl;
            cv::imwrite("captured_frame.png", frame);
        } else if (key == 'o') {
            lastKeypress = 'o';  // update last keypress variable
            cout << lastKeypress << " pressed: Converting to original colors." << endl;
        } else if (key == 'g') {
            lastKeypress = 'g';
            cout << lastKeypress << " pressed: Converting to opencv greyscale." << endl;
        } else if (key == 'h') {
            lastKeypress = 'h';
            cout << lastKeypress << " pressed: Converting to custom greyscale." << endl;
        } else if (key == 't') {
            lastKeypress = 't';
            cout << lastKeypress << " pressed: Converting to custom sepia." << endl;
        } else if (key == 'b') {
            lastKeypress = 'b';
            cout << lastKeypress << " pressed: Converting to custom blur." << endl;
        } else if (key == 'B') {
            lastKeypress = 'B';
            cout << lastKeypress << " pressed: Converting to custom (faster) blur." << endl;
        } else if (key == 'x') {
            lastKeypress = 'x';
            cout << lastKeypress << "pressed : Converting to sobelX filter." << endl;
        } else if (key == 'y') {
            lastKeypress = 'y';
            cout << lastKeypress << "pressed : Converting to sobelY filter." << endl;
        }

    }

    return 0;
}
