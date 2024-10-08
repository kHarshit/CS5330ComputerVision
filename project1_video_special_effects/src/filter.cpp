/**
 * author: Harshit Kumar and Khushi Neema
 * date: Jan 20, 2024
 * purpose: Applies a filter to an image.
 *
 */

#include "filter.h"

int greyscale(cv::Mat &src, cv::Mat &dst)
{
    // check src and dst Mat consistency
    if (src.size() != dst.size())
    {
        return -1;
    }

    // apply custom greyscale transformation
    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            // subtract the red channel value from 255
            int greyValue = 255 - src.at<cv::Vec3b>(i, j)[2];
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(greyValue, greyValue, greyValue);
        }
    }

    return 0;
}

int sepia(cv::Mat &src, cv::Mat &dst)
{
    // check src and dst Mat consistency
    if (src.size() != dst.size())
    {
        return -1;
    }

    // apply sepia transformation
    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            // R .    G .    B .
            // 0.272, 0.534, 0.131    // Blue coefficients
            // 0.349, 0.686, 0.168    // Green coefficients
            // 0.393, 0.769, 0.189     // Red coefficients
            int blue = (src.at<cv::Vec3b>(i, j)[2] * 0.272) + (src.at<cv::Vec3b>(i, j)[1] * 0.534) + (src.at<cv::Vec3b>(i, j)[0] * 0.131);
            int green = (src.at<cv::Vec3b>(i, j)[2] * 0.349) + (src.at<cv::Vec3b>(i, j)[1] * 0.686) + (src.at<cv::Vec3b>(i, j)[0] * 0.168);
            int red = (src.at<cv::Vec3b>(i, j)[2] * 0.393) + (src.at<cv::Vec3b>(i, j)[1] * 0.769) + (src.at<cv::Vec3b>(i, j)[0] * 0.189);
            // clip values to [0, 255]
            blue = blue > 255 ? 255 : blue;
            green = green > 255 ? 255 : green;
            red = red > 255 ? 255 : red;
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green, red);
        }
    }

    return 0;
}

int blur5x5_1(cv::Mat &src, cv::Mat &dst)
{
    dst = src.clone();

    // Gaussian kernel
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}};

    // Loop through each pixel, excluding the first two and last two rows and columns
    for (int i = 2; i < src.rows - 2; ++i)
    {
        for (int j = 2; j < src.cols - 2; ++j)
        {
            // Separate channels
            for (int c = 0; c < src.channels(); ++c)
            {
                int sum = 0;
                // Apply the blur filter
                for (int m = -2; m <= 2; ++m)
                {
                    for (int n = -2; n <= 2; ++n)
                    {
                        sum += src.at<cv::Vec3b>(i + m, j + n)[c] * kernel[m + 2][n + 2];
                    }
                }
                // Normalize and set the pixel value in the destination image
                // dst.at<cv::Vec3b>(i, j)[c] = <cv::uchar>(sum / 84);
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

int blur5x5_2(cv::Mat &src, cv::Mat &dst)
{
    dst = cv::Mat::zeros(src.size(), src.type());
    cv::Mat temp = cv::Mat::zeros(src.size(), src.type());

    // row filter
    for (int i = 2; i < src.rows - 2; i++)
    {
        cv::Vec3b *rowptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *tempptr = temp.ptr<cv::Vec3b>(i);

        // loop over columns
        for (int j = 2; j < src.cols - 2; j++)
        {
            // loop over color channels
            for (int c = 0; c < 3; c++)
            {
                // row filter [1 , 2, 4, 2, 1]
                tempptr[j][c] = (1 * rowptr[j - 2][c] + 2 * rowptr[j - 1][c] + 4 * rowptr[j][c] + 2 * rowptr[j + 1][c] + 1 * rowptr[j + 2][c]) / 10.0;
            }
        }
    }

    // column filter
    for (int i = 2; i < src.rows - 2; i++)
    {
        cv::Vec3b *rowptrm2 = temp.ptr<cv::Vec3b>(i - 2);
        cv::Vec3b *rowptrm1 = temp.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b *rowptr = temp.ptr<cv::Vec3b>(i);
        cv::Vec3b *rowptrp1 = temp.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b *rowptrp2 = temp.ptr<cv::Vec3b>(i + 2);

        cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);

        // loop over columns
        for (int j = 2; j < src.cols - 2; j++)
        {
            // loop over color channels
            for (int c = 0; c < 3; c++)
            {
                /*column filter
                    [1]  m2
                    [2]  m1
                    [4]  r
                    [2]  p1
                    [1]  p2
                */
                dptr[j][c] = (1 * rowptrm2[j][c] + 2 * rowptrm1[j][c] + 4 * rowptr[j][c] + 2 * rowptrp1[j][c] + 1 * rowptrp2[j][c]) / 10.0;
                // clip b/w 0 and 255
                dptr[j][c] = dptr[j][c] > 255 ? 255 : dptr[j][c];
            }
        }
    }

    return 0; // Success
}

#if 0
int blur5x5_2(cv::Mat &src, cv::Mat &dst)
{
    //NOTE: This function is not used in the final project (check uncommented one above)
    dst = cv::Mat::zeros(src.size(), src.type());

    // loop over rows
    for (int i = 2; i < src.rows - 2; i++)
    {

        // src row pointers
        cv::Vec3b *rptrm2 = src.ptr<cv::Vec3b>(i - 2);
        cv::Vec3b *rptrm1 = src.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *rptrp1 = src.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b *rptrp2 = src.ptr<cv::Vec3b>(i + 2);

        // destination ptr
        cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);

        // loop over columns
        for (int j = 2; j < src.cols - 2; j++)
        {

            // loop over color channels
            for (int c = 0; c < 3; c++)
            {

                // row filter [1 , 2, 4, 2, 1]
                dptr[j][c] = (1 * rptr[j - 2][c] + 2 * rptr[j - 1][c] + 4 * rptr[j][c] + 2 * rptr[j + 1][c] + 1 * rptr[j + 2][c]) / 10.0;

                /*column filter
                    [1]  m2
                    [2]  m1
                    [4]  r
                    [2]  p1
                    [1]  p2
                */

                dptr[j][c] = (dptr[j][c] + 1 * rptrm2[j][c] + 2 * rptrm1[j][c] + 4 * rptr[j][c] + 2 * rptrp1[j][c] + 1 * rptrp2[j][c]) / 10.0;
                // clip b/w 0 and 255
                // dptr[j][c] = dptr[j][c] > 255 ? 255 : dptr[j][c];
            }
        }
    }

    return 0; // Success
}

int sobelX3x3(cv::Mat& src, cv::Mat& dst) {
    // NOTE: This function is not used in the final project (check uncommented one below)
    //X Sobel
    //[-1,0,1],     [1] [-1,0,1]
    //[-2,0,2], ==> [2]X
    //[-1,0,1]      [1]
    dst = cv::Mat::zeros(src.size(), CV_16SC3);
    for (int i = 1; i < src.rows - 1; i++) {
        cv::Vec3b* rptrm1 = src.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b* rptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* rptrp1 = src.ptr<cv::Vec3b>(i + 1);

        cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);
        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                //row filter
                dptr[j][c] = -1 * rptr[j - 1][c] + 1 * rptr[j + 1][c];

                dptr[j][c] = (dptr[j][c] + (1 * rptrm1[j][c] + 2 * rptr[j][c] + 1 * rptrp1[j][c])) / 4;
            }
        }
    }

    return 0;
}

int sobelY3x3(cv::Mat& src, cv::Mat& dst) {
    // NOTE: This function is not used in the final project (check uncommented one below)
    //Y Sobel
    //[-1,2,1],        [1]    [1 2 1] 
    //[0,0,0], ====>   [0]  X 
    //[-1,-2,-1]       [-1]
    dst = cv::Mat::zeros(src.size(), CV_16SC3);
    for (int i = 1; i < src.rows - 1; i++) {
        cv::Vec3b* rptrm1 = src.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b* rptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* rptrp1 = src.ptr<cv::Vec3b>(i + 1);

        cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);
        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                //row filter
                dptr[j][c] = 1 * rptr[j - 1][c] + -1 * rptr[j + 1][c];

                dptr[j][c] = (dptr[j][c] + (1 * rptrm1[j][c] + 2 * rptr[j][c] + 1 * rptrp1[j][c])) / 4;
            }
        }
    }

    return 0;
}
#endif

int sobelX3x3(cv::Mat &src, cv::Mat &dst)
{
    cv::Mat temp = src.clone();
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    // horizontal filter 
    for (int i = 0; i < src.rows; i++)
    {
        cv::Vec3b *tempptr = temp.ptr<cv::Vec3b>(i);
        cv::Vec3b *rowptr = src.ptr<cv::Vec3b>(i);
        for (int j = 1; j < src.cols - 1; j++)
        {
            for (int c = 0; c < 3; c++)
            {
                tempptr[j][c] = static_cast<uchar>(
                    (-1 * rowptr[j - 1][c] +
                     1 * rowptr[j + 1][c]) /
                        2.0 +
                    0.5);
            }
        }
    }

    // vertical filter
    for (int i = 1; i < src.rows - 1; i++)
    {
        cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);
        cv::Vec3b *tempptrm1 = temp.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b *tempptr = temp.ptr<cv::Vec3b>(i);
        cv::Vec3b *tempptrp1 = temp.ptr<cv::Vec3b>(i + 1);
        for (int j = 0; j < src.cols; j++)
        {
            for (int c = 0; c < 3; c++)
            {
                dptr[j][c] = static_cast<short>(
                    (1 * tempptrm1[j][c] +
                     2 * tempptr[j][c] +
                     1 * tempptrp1[j][c]) /
                        4.0 +
                    0.5);
            }
        }
    }

    return 0;
}

int sobelY3x3(cv::Mat &src, cv::Mat &dst)
{
    cv::Mat temp = src.clone();
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    // vertical filter
    for (int i = 1; i < src.rows - 1; i++)
    {
        cv::Vec3b *tempptr = temp.ptr<cv::Vec3b>(i);
        cv::Vec3b *rowptrm1 = src.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b *rowptrp1 = src.ptr<cv::Vec3b>(i + 1);
        for (int j = 0; j < src.cols; j++)
        {
            for (int c = 0; c < 3; c++)
            {
                tempptr[j][c] = static_cast<uchar>(
                    (-1 * rowptrm1[j][c] +
                     1 * rowptrp1[j][c]) /
                        2.0 +
                    0.5);
            }
        }
    }

    // horizontal filter
    for (int i = 0; i < src.rows; i++)
    {
        cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);
        cv::Vec3b *tempptr = temp.ptr<cv::Vec3b>(i);
        for (int j = 1; j < src.cols - 1; j++)
        {
            for (int c = 0; c < 3; c++)
            {
                dptr[j][c] = static_cast<short>(
                    (1 * tempptr[j - 1][c] +
                     2 * tempptr[j][c] +
                     1 * tempptr[j + 1][c]) /
                        4.0 +
                    0.5);
            }
        }
    }

    return 0;
}

int magnitude(cv::Mat &sobelX, cv::Mat &sobelY, cv::Mat &dst)
{
    dst = cv::Mat::zeros(sobelX.size(), CV_8UC3);

    // loop over columns
    for (int i = 0; i < sobelX.rows; i++)
    {

        // src row pointers
        cv::Vec3s *sobelXrowptr = sobelX.ptr<cv::Vec3s>(i);
        cv::Vec3s *sobelYrowptr = sobelY.ptr<cv::Vec3s>(i);

        // destination ptr
        cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);

        // loop over columns
        for (int j = 0; j < sobelX.cols; j++)
        {
            // loop over color channels
            for (int c = 0; c < 3; c++)
            {
                dptr[j][c] = sqrt((sobelXrowptr[j][c] * sobelXrowptr[j][c] + sobelYrowptr[j][c] * sobelYrowptr[j][c]));
            }
        }
    }

    return 0;
}

int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels)
{
    dst = cv::Mat::zeros(src.size(), src.type());

    // create a temporary image
    cv::Mat tempImg;
    tempImg = cv::Mat::zeros(src.size(), src.type());

    // calculate the quantization factor
    int b;
    b = 255 / levels;

    cv::Mat x;
    blur5x5_2(src, x);

    for (int i = 0; i < x.rows; i++)
    {
        cv::Vec3b *xrowptr = x.ptr<cv::Vec3b>(i);
        cv::Vec3b *tempptr = tempImg.ptr<cv::Vec3b>(i);
        cv::Vec3b *drowptr = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < x.cols; j++)
        {
            for (int c = 0; c < 3; c++)
            {
                tempptr[j][c] = xrowptr[j][c] / b;
                drowptr[j][c] = tempptr[j][c] * b;
            }
        }
    }

    return 0;
}

int comicBookEffect(cv::Mat &input, cv::Mat &output)
{
    // bilateral filter for smoothing while preserving edges
    cv::Mat bilateralFiltered;
    cv::bilateralFilter(input, bilateralFiltered, 9, 75, 75);

    cv::Mat gray, thresholded, edges;
    cv::cvtColor(bilateralFiltered, gray, cv::COLOR_BGR2GRAY);

    // adaptive thresholding
    cv::adaptiveThreshold(gray, thresholded, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 9, 9);

    // edge enhancement
    cv::Canny(gray, edges, 50, 150);
    cv::bitwise_and(thresholded, edges, output);

    return 0;
}
