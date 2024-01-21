/**
 * author: Harshit Kumar
 * date: Jan 20, 2024
 * purpose: Applies a filter to an image.
 * 
*/

#include "filter.h"

int greyscale(cv::Mat& src, cv::Mat& dst) {
    // check src and dst Mat consistency
    if (src.size() != dst.size()) {
        return -1;
    }

    // apply custom greyscale transformation
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            // Subtract the red channel value from 255
            int greyValue = 255 - src.at<cv::Vec3b>(i, j)[2];
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(greyValue, greyValue, greyValue);
        }
    }

    return 0;
}
