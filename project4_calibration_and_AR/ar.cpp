#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

void drawCube(const Mat& img, const vector<Point2f>& pts) {
    // Assuming pts[0] is the bottom-left corner of the target
    Point2f base = pts[0]; // Base point for drawing the cube

    int size = 600; // Size of the cube
    // Points for base of the cube
    vector<Point> basePoints = {
        Point(base.x, base.y),
        Point(base.x + size, base.y),
        Point(base.x + size, base.y + size),
        Point(base.x, base.y + size)
    };
    
    // Points for top of the cube
    vector<Point> topPoints = {
        Point(base.x, base.y - size),
        Point(base.x + size, base.y - size),
        Point(base.x + size, base.y - size + size),
        Point(base.x, base.y - size + size)
    };
    
    // Draw base
    for(int i = 0; i < 4; i++) {
        line(img, basePoints[i], basePoints[(i+1)%4], Scalar(255, 0, 0), 2);
    }

    // Draw top
    for(int i = 0; i < 4; i++) {
        line(img, topPoints[i], topPoints[(i+1)%4], Scalar(255, 0, 0), 2);
    }

    // Draw pillars
    for(int i = 0; i < 4; i++) {
        line(img, basePoints[i], topPoints[i], Scalar(255, 0, 0), 2);
    }
}


#if 0
void drawCube(const Mat& img, const vector<Point2f>& pts) {
    vector<Point3f> pts3D;
    pts3D.push_back(Point3f(0, 0, 0));
    pts3D.push_back(Point3f(1, 0, 0));
    pts3D.push_back(Point3f(1, 1, 0));
    pts3D.push_back(Point3f(0, 1, 0));
    pts3D.push_back(Point3f(0, 0, 1));
    pts3D.push_back(Point3f(1, 0, 1));
    pts3D.push_back(Point3f(1, 1, 1));
    pts3D.push_back(Point3f(0, 1, 1));

    for (int i = 0; i < 4; i++) {
        line(img, pts[i], pts[(i+1)%4], Scalar(0, 255, 0), 2);
        line(img, pts[i+4], pts[(i+1)%4+4], Scalar(0, 255, 0), 2);
        line(img, pts[i], pts[i+4], Scalar(0, 255, 0), 2);
    }
}
#endif

// Function to detect and draw anything you want on the detected target.
void augmentReality(const Mat& frame, const vector<Point2f>& pts) {
    // Example: Drawing a simple quadrilateral around the detected target.
    for (size_t i = 0; i < pts.size(); i++) {
        line(frame, pts[i], pts[(i+1) % pts.size()], Scalar(0, 255, 0), 4);
    }
    // You can extend this function to draw 3D objects, text, or images.
    if (pts.size() >= 4)
    {
        cout << "Drawing a cube on the detected target." << endl;
        // Draw a cube on the detected target.
        drawCube(frame, pts);
    }
}


int main() {
    // Load the target image (the non-checkerboard target you want to detect)
    Mat targetImage = imread("/Users/harshit/Downloads/IMG_2372.jpeg", IMREAD_GRAYSCALE);
    if (targetImage.empty()) {
        cout << "Error loading target image." << endl;
        return -1;
    }

    // Initialize the ORB detector
    // Ptr<ORB> orb = ORB::create();
    // Increase the number of features and adjust other parameters as needed
    Ptr<ORB> orb = ORB::create(10000, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);


    vector<KeyPoint> keypointsTarget;
    Mat descriptorsTarget;
    orb->detectAndCompute(targetImage, noArray(), keypointsTarget, descriptorsTarget);

    // Initialize the matcher
    BFMatcher matcher(NORM_HAMMING);

    // Start capturing video
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error opening video stream." << endl;
        return -1;
    }

    Mat frame, grayFrame;
    vector<KeyPoint> keypointsFrame;
    Mat descriptorsFrame;

    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        // Convert frame to grayscale because ORB works with grayscale images
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // Detect ORB features in the frame
        orb->detectAndCompute(grayFrame, noArray(), keypointsFrame, descriptorsFrame);

        // Match the features from the target image to the current frame
        vector<vector<DMatch>> matches;
        matcher.knnMatch(descriptorsTarget, descriptorsFrame, matches, 2);

        // Filter matches using the Lowe's ratio test
        float ratioThresh = 0.75;
        vector<DMatch> goodMatches;
        for (size_t i = 0; i < matches.size(); i++) {
            if (matches[i][0].distance < ratioThresh * matches[i][1].distance) {
                goodMatches.push_back(matches[i][0]);
            }
        }

        cout << "Matches Found: " << matches.size() << ", Good Matches: " << goodMatches.size() << endl;

        // If enough matches are found, identify the target object's location in the frame
        if (goodMatches.size() >= 60) {
            // Extract location of good matches
            vector<Point2f> pointsTarget, pointsFrame;
            for (size_t i = 0; i < goodMatches.size(); i++) {
                pointsTarget.push_back(keypointsTarget[goodMatches[i].queryIdx].pt);
                pointsFrame.push_back(keypointsFrame[goodMatches[i].trainIdx].pt);
            }

            // Find homography
            Mat H = findHomography(pointsTarget, pointsFrame, RANSAC);

            // Check if the homography matrix is empty
            if (!H.empty()) {
                // Use homography to project the corners of the target image onto the video frame
                vector<Point2f> cornersTarget(4);
                cornersTarget[0] = Point2f(0, 0);
                cornersTarget[1] = Point2f((float)targetImage.cols, 0);
                cornersTarget[2] = Point2f((float)targetImage.cols, (float)targetImage.rows);
                cornersTarget[3] = Point2f(0, (float)targetImage.rows);
                vector<Point2f> cornersFrame(4);

                perspectiveTransform(cornersTarget, cornersFrame, H);

                // Augment the reality based on the corners
                augmentReality(frame, cornersFrame);
            }
        }

        imshow("Augmented Reality", frame);

        if (waitKey(5) >= 0)
            break;
    }

    return 0;
}
