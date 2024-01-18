/**
 * author: Harshit Kumar
 * date: Jan 18, 2024
 * purpose: Display an image and exit when the user presses 'q'.
 * 
*/

#include "imgDisplay.h"

using namespace cv;

int displayImage(const std::string& imgPath) {
    // read input image
    Mat img = imread(imgPath, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << imgPath << std::endl;
        return -1;
    }
    imshow("Display window", img);

    while (true) {
        // wait for keypress
        int key = cv::waitKey(0);

        // exit on q
        if (key == 'q') {
            break;
        }
    }

    // close window
    cv::destroyAllWindows();

    return 0;
}
