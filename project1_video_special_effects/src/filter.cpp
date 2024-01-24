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
int blur5x5_1(cv::Mat& src, cv::Mat& dst) {
    dst = src.clone();

    // Gaussian kernel
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    // Loop through each pixel, excluding the first two and last two rows and columns
    for (int i = 2; i < src.rows - 2; ++i) {
        for (int j = 2; j < src.cols - 2; ++j) {
            // Separate channels
            for (int c = 0; c < src.channels(); ++c) {
                int sum = 0;
                // Apply the blur filter
                for (int m = -2; m <= 2; ++m) {
                    for (int n = -2; n <= 2; ++n) {
                        sum += src.at<cv::Vec3b>(i + m, j + n)[c] * kernel[m + 2][n + 2];
                    }
                }
                // Normalize and set the pixel value in the destination image
                //dst.at<cv::Vec3b>(i, j)[c] = <cv::uchar>(sum / 84); 
                float normalizedValue = static_cast<float>(sum) / 84.0; // 84 is the sum of the kernel values
                // Clip the value to the valid range [0, 255]
                if (normalizedValue < 0.0)
                    normalizedValue = 0.0;
                else if (normalizedValue > 255.0)
                    normalizedValue = 255.0;

                // Set the pixel value in the destination image
                dst.at<cv::Vec3b>(i, j)[c] = static_cast<uchar>(normalizedValue);
            }
        }
    }
    return 0;
}




int blur5x5_2(cv::Mat & src, cv::Mat & dst) {

    return 0;
}

