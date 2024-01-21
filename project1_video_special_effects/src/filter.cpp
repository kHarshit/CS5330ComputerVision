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
            // subtract the red channel value from 255
            int greyValue = 255 - src.at<cv::Vec3b>(i, j)[2];
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(greyValue, greyValue, greyValue);
        }
    }

    return 0;
}

int sepia(cv::Mat &src, cv::Mat &dst) {
    // check src and dst Mat consistency
    if (src.size() != dst.size()) {
        return -1;
    }

    // apply sepia transformation
    for (int i=0; i<src.rows; ++i) {
        for (int j=0; j<src.cols; ++j) {
            // R .    G .    B .    
            // 0.272, 0.534, 0.131    // Red coefficients
            // 0.349, 0.686, 0.168    // Green coefficients
            // 0.393, 0.769, 0.189     // Blue coefficients
        int red = (src.at<cv::Vec3b>(i, j)[2] * 0.272) + (src.at<cv::Vec3b>(i, j)[1] * 0.534) + (src.at<cv::Vec3b>(i, j)[0] * 0.131);
        int green = (src.at<cv::Vec3b>(i, j)[2] * 0.349) + (src.at<cv::Vec3b>(i, j)[1] * 0.686) + (src.at<cv::Vec3b>(i, j)[0] * 0.168);
        int blue = (src.at<cv::Vec3b>(i, j)[2] * 0.393) + (src.at<cv::Vec3b>(i, j)[1] * 0.769) + (src.at<cv::Vec3b>(i, j)[0] * 0.189);
        dst.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green, red);
        }
    }


    return 0;
}

int blur5x5_1( cv::Mat &src, cv::Mat &dst ) {

    return 0;
}


int blur5x5_2( cv::Mat &src, cv::Mat &dst ) {

    return 0;
}
