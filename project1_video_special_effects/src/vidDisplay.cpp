/**
 * author: Harshit Kumar and Khushi Neema
 * date: Jan 18, 2024
 * purpose: Display video, allowing users to save frames and quit using keystrokes.
 *
*/

#include <iostream>
#include "vidDisplay.h"
#include "filter.h"
#include "faceDetect.h"

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
    cv::Mat grey;
    std::vector<cv::Rect> faces;
    cv::Rect last(0, 0, 0, 0);
    cv::VideoWriter video;
    bool isVideoWriterInitialized = false;
    bool isSavingVideo = false;

    char lastKeypress = '\0';  // Initialize the last keypress variable

    for (;;) {
        capdev >> frame; // Get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("Frame is empty\n");
            break;
        }

        if (!isVideoWriterInitialized) {
            video.open("out.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frame.cols, frame.rows));
            isVideoWriterInitialized = true;
        }

        // Check the last keypress and modify the image accordingly
        if (lastKeypress == 'g') {
            cv::putText(frame, "OpenCV Greyscale", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        }
        else if (lastKeypress == 'h') {
            // Use the custom greyscale function
            cv::putText(frame, "Custom Greyscale", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
            greyscale(frame, frame);
        }
        else if (lastKeypress == 't') {
            // Use the custom sepia function
            cv::putText(frame, "Custom Sepia", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
            sepia(frame, frame);
        }
        else if (lastKeypress == 'b') {
            // Use the custom blur function
            cv::putText(frame, "Custom Blur", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
            blur5x5_1(frame, frame);
        }
        else if (lastKeypress == 'B') {
            // Use the custom (faster) blur function
            cv::putText(frame, "Custom Blur (faster)", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
            blur5x5_2(frame, filter);
            cv::convertScaleAbs(filter, frame);
        }
        else if (lastKeypress == 'x') {
            cv::putText(frame, "SobelX", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
            sobelX3x3(frame, filter);
            cv::convertScaleAbs(filter, frame);
            // cv::imshow("SobelX", filter);
        }
        else if (lastKeypress == 'y') {
            cv::putText(frame, "SobelY", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
            sobelY3x3(frame, filter);
            cv::convertScaleAbs(filter, frame);
            // cv::imshow("SobelY", filter);
        }
        else if (lastKeypress == 'm') {
            cv::putText(frame, "Gradient Image from Sobel X and Y", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 255, 0), 3);
            cv::Mat grad_x, grad_y;
            sobelX3x3(frame, grad_x);
            sobelY3x3(frame, grad_y);
            //Calling Gradient Function 
            magnitude(grad_x, grad_y, filter);
            cv::convertScaleAbs(filter, frame);
        }
        else if (lastKeypress == 'l') {
            cv::putText(frame, "Blur and quantize", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 255, 0), 3);
            blurQuantize(frame, filter, 10);
            cv::convertScaleAbs(filter, frame);
        }
        else if (lastKeypress == 'f') {
            cv::putText(frame, "Face Detect", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
            // use facedetect
            // convert the image to greyscale
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY, 0);
            // detect faces
            detectFaces(grey, faces);
            // draw boxes around the faces
            drawBoxes(frame, faces);
            // add a little smoothing by averaging the last two detections
            if (faces.size() > 0) {
                last.x = (faces[0].x + last.x) / 2;
                last.y = (faces[0].y + last.y) / 2;
                last.width = (faces[0].width + last.width) / 2;
                last.height = (faces[0].height + last.height) / 2;
            }
        }
        else if (lastKeypress == 'c') {
            cv::putText(frame, "Colorful Face", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
            // Make the face colorful, while the rest of the image is greyscale.
            // use facedetect
            // convert the image to greyscale
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY, 0);
            // detect faces
            detectFaces(grey, faces);
            // draw boxes around the faces
            drawBoxes(frame, faces);
            // add a little smoothing by averaging the last two detections
            if (faces.size() > 0) {
                last.x = (faces[0].x + last.x) / 2;
                last.y = (faces[0].y + last.y) / 2;
                last.width = (faces[0].width + last.width) / 2;
                last.height = (faces[0].height + last.height) / 2;
            }
            // Create a grayscale version of the frame
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray, gray, cv::COLOR_GRAY2BGR);

            cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8U);
            for (const auto& face : faces) {
                // Include faces in the mask
                cv::rectangle(mask, face, cv::Scalar(255), -1);
            }
            cv::Mat mask3;
            cv::cvtColor(mask, mask3, cv::COLOR_GRAY2BGR);
            frame = (frame & mask3) + (gray & ~mask3);
        }
        else if (lastKeypress == 'n') {
            cv::putText(frame, "Negative Image", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
            // negative image
            frame = 255 - frame;
        }
        else if (lastKeypress == 'e') {
            cv::putText(frame, "Embossing Effect", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 255, 0), 3);
            // Make an embossing effect
            cv::Mat grad_x, grad_y;
            sobelX3x3(frame, grad_x);
            sobelY3x3(frame, grad_y);
            cv::convertScaleAbs(grad_x, grad_x);
            cv::convertScaleAbs(grad_y, grad_y);
            frame = grad_x * 0.7071 + grad_y * 0.7071;
        }
        
        else if (lastKeypress == 'z') {
            // comic book effect
            cv::putText(frame, "Comic Book Effect", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(255, 0, 255), 3);
            comicBookEffect(frame, frame);
        }
        else if (lastKeypress == 'v') {
            cv::putText(frame, "Video Saving Started", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(255, 0, 0), 3);
            // save short video sequences with the special effects
            // save the current frame to the video
            isSavingVideo = true;
        }
        else if (lastKeypress == '0') {
            cv::putText(frame, "Video Saving Stopped", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(255, 0, 0), 3);
            // stop video saving
            isSavingVideo = false;
        }
        else {
            cv::putText(frame, "Original Video", cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(255, 0, 0), 3);
        }

        if (isSavingVideo) {
            video.write(frame);
        }

        cv::imshow("Video", frame);

        // check waiting keystroke
        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        }
        else if (key == 's') {
            cout << key << " pressed: Saving frame to captured_frame.png." << endl;
            cv::imwrite("captured_frame.png", frame);
        }
        else if (key == 'o') {
            lastKeypress = 'o';  // update last keypress variable
            cout << lastKeypress << " pressed: Converting to original colors." << endl;
        }
        else if (key == 'g') {
            lastKeypress = 'g';
            cout << lastKeypress << " pressed: Converting to opencv greyscale." << endl;
        }
        else if (key == 'h') {
            lastKeypress = 'h';
            cout << lastKeypress << " pressed: Converting to custom greyscale." << endl;
        }
        else if (key == 't') {
            lastKeypress = 't';
            cout << lastKeypress << " pressed: Converting to custom sepia." << endl;
        }
        else if (key == 'b') {
            lastKeypress = 'b';
            cout << lastKeypress << " pressed: Converting to custom blur." << endl;
        }
        else if (key == 'B') {
            lastKeypress = 'B';
            cout << lastKeypress << " pressed: Converting to custom (faster) blur." << endl;
        }
        else if (key == 'x') {
            lastKeypress = 'x';
            cout << lastKeypress << "pressed : Converting to sobelX filter." << endl;
        }
        else if (key == 'y') {
            lastKeypress = 'y';
            cout << lastKeypress << "pressed : Converting to sobelY filter." << endl;
        }
        else if (key == 'l') {
            lastKeypress = 'l';
            cout << lastKeypress << "pressed : Blur and Quantizing Image" << endl;
        }
        else if (key == 'm') {
            lastKeypress = 'm';
            cout << lastKeypress << "pressed : Generating Gradient Magnitude Image" << endl;
        }
        else if (key == 'f') {
            lastKeypress = 'f';
            cout << lastKeypress << "pressed : Using face detect." << endl;
        }
        else if (key == 'a') {
            lastKeypress = 'a';
            cout << lastKeypress << "pressed : Make background greyscale." << endl;
        }
        else if (key == 'c') {
            lastKeypress = 'c';
            cout << lastKeypress << "pressed : Make face colorful." << endl;
        }
        else if (key == 'n') {
            lastKeypress = 'n';
            cout << lastKeypress << "pressed : Make negative image." << endl;
        }
        else if (key == 'e') {
            lastKeypress = 'e';
            cout << lastKeypress << "pressed : Make embossing effect." << endl;
        }
        else if (key == 'z') {
            lastKeypress = 'z';
            cout << lastKeypress << "pressed : Make comic book effect." << endl;
        }
        else if (key == 'v') {
            lastKeypress = 'v';
            cout << lastKeypress << "pressed : Video saving started." << endl;
        }
        else if (key == '0') {
            lastKeypress = '0';
            cout << lastKeypress << "pressed : Video saving stopped." << endl;
        }
    }

    return 0;
}
