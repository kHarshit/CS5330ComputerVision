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
        }

        cv::imshow("Video", frame);

        // check waiting keystroke
        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        } else if (key == 's') {
            cv::imwrite("captured_frame.png", frame);
        } else if (key == 'g') {
            lastKeypress = 'g';  // update last keypress variable
            cout << lastKeypress << " pressed: Converting to opencv greyscale." << endl;
        } else if (key == 'h') {
            lastKeypress = 'h';  // update last keypress variable
            cout << lastKeypress << " pressed: Converting to custom greyscale." << endl;
        }
    }

    return 0;
}
