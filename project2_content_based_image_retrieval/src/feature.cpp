/**
 * author: Harshit Kumar, Khushi Neema
 * date: Jan 29, 2024
 * purpose: Implements various image processing features wrt Content-based Image Retrieval
 * such as baseline matching, histogram matching, etc.
 *
 */

#include "distance.h"
#include "feature.h"
#include <iostream>

/**
 * Convert a cv::Mat to a 1D vector of floats
 * @param m Input cv::Mat
 * @return std::vector<float> 1D vector of floats
 */
std::vector<float> matToVector(const cv::Mat &m)
{
    // cv::Mat flat = m.reshape(1, m.total() * m.channels());
    cv::Mat flat;
    if (m.isContinuous())
    {
        flat = m.reshape(1, m.total() * m.channels());
    }
    else
    {
        cv::Mat continuousM;
        m.copyTo(continuousM);
        flat = continuousM.reshape(1, continuousM.total() * continuousM.channels());
    }
    flat.convertTo(flat, CV_32F);
    return m.isContinuous() ? flat : flat.clone();
}

std::vector<float> computeBaselineFeatures(const cv::Mat &image)
{
    // Ensure the image is large enough for a 7x7 extraction
    if (image.cols < 7 || image.rows < 7)
    {
        throw std::runtime_error("Image is too small for a 7x7 feature extraction.");
    }
    int startX = (image.cols - 7) / 2;
    int startY = (image.rows - 7) / 2;
    return matToVector(image(cv::Rect(startX, startY, 7, 7)));
}

cv::Mat computeRGChromaticityHistogram(const cv::Mat &image, int bins)
{
    cv::Mat histogram = cv::Mat::zeros(bins, bins, CV_32F);

    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            float R = pixel[2];
            float G = pixel[1];
            float B = pixel[0];
            float sum = R + G + B;

            if (sum > 0)
            { // Avoid division by zero
                float r = R / sum;
                float g = G / sum;

                // calculate bin indices for R and G
                int r_bin = static_cast<int>(r * (bins - 1) + 0.5);
                int g_bin = static_cast<int>(g * (bins - 1) + 0.5);

                histogram.at<float>(r_bin, g_bin) += 1.0f;
            }
        }
    }

    // Compute total number of pixels to normalize histogram
    float totalPixels = image.rows * image.cols;

    // Normalize the histogram so that the sum of histogram bins = 1
    histogram /= totalPixels;

    return histogram;
}

cv::Mat computeRGBHistogram(const cv::Mat &image, int bins)
{
    // Initialize a 3D histogram with given bins for each dimension and float type
    // Define the size for each dimension of the histogram
    int histSize[] = {bins, bins, bins};
    // Create a 3D histogram with floating-point values
    cv::Mat histogram(3, histSize, CV_32F, cv::Scalar(0));

    // cv::Mat histogram = cv::Mat::zeros(bins, bins, bins, CV_32F);

    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            float R = pixel[2];
            float G = pixel[1];
            float B = pixel[0];

            // Calculate bin indices for R, G, and B
            int r_bin = static_cast<int>(R * (bins - 1) / 255.0 + 0.5);
            int g_bin = static_cast<int>(G * (bins - 1) / 255.0 + 0.5);
            int b_bin = static_cast<int>(B * (bins - 1) / 255.0 + 0.5);

            // Increment the corresponding bin
            *histogram.ptr<float>(r_bin, g_bin, b_bin) += 1.0f;
        }
    }

    // Compute total number of pixels to normalize histogram
    float totalPixels = image.rows * image.cols;

    // Normalize the histogram so that the sum of histogram bins = 1
    histogram /= totalPixels;

    return histogram;
}

std::pair<cv::Mat, cv::Mat> computeSpatialHistograms(const cv::Mat &image, int bins)
{
    // Split image into top and bottom halves
    cv::Mat topHalf = image(cv::Rect(0, 0, image.cols, image.rows / 2));
    cv::Mat bottomHalf = image(cv::Rect(0, image.rows / 2, image.cols, image.rows / 2));

    // Compute RGB histograms for each part
    cv::Mat topHist = computeRGBHistogram(topHalf, bins);
    cv::Mat bottomHist = computeRGBHistogram(bottomHalf, bins);

    return {topHist, bottomHist};
}

cv::Mat computeGrassChromaticityHistogram(const cv::Mat &image, int bins)
{
    cv::Mat histogram = cv::Mat::zeros(bins, bins, CV_32F);

    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            float G = pixel[1];
            float B = pixel[0];
            float sum = G + B;

            if (sum > 0)
            { // Avoid division by zero
                float g = G / sum;
                float b = B / sum;

                // Calculate bin indices for G and B
                int g_bin = static_cast<int>(g * (bins - 1) + 0.5);
                int b_bin = static_cast<int>(b * (bins - 1) + 0.5);

                // Increment the histogram bin for green-blue chromaticity
                histogram.at<float>(g_bin, b_bin) += 1.0f;
            }
        }
    }

    // Normalize the histogram so that the sum of histogram bins = 1
    cv::normalize(histogram, histogram, 1, 0, cv::NORM_L1);

    return histogram;
}

cv::Mat computeBlueChromaticityHistogram(const cv::Mat &image, int bins)
{
    cv::Mat histogram = cv::Mat::zeros(bins, bins, CV_32F);

    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            float R = pixel[2];
            float G = pixel[1];
            float B = pixel[0];
            float sum = R + G + B;

            if (sum > 0)
            { // Avoid division by zero
                // Adjust chromaticity computation to emphasize blue
                float b = B / sum;
                float rg = (R + G) / sum; // Combined chromaticity of R and G to contrast with B

                // Calculate bin indices for B and RG (using RG as a contrast to B)
                int b_bin = static_cast<int>(b * (bins - 1) + 0.5);
                int rg_bin = static_cast<int>(rg * (bins - 1) + 0.5);

                // Increment the histogram bin for blue chromaticity
                histogram.at<float>(b_bin, rg_bin) += 1.0f;
            }
        }
    }

    // Normalize the histogram so that the sum of histogram bins = 1
    cv::normalize(histogram, histogram, 1, 0, cv::NORM_L1);

    return histogram;
}

double computeEdgeDensity(const cv::Mat &image)
{
    cv::Mat gray, edges;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 100, 200); // params may need adjustment

    return cv::sum(edges)[0] / (edges.rows * edges.cols);
}

double computeGrassCoverage(const cv::Mat &image)
{
    // compute the amount of green in the lower half of the image
    cv::Mat lowerHalf = image(cv::Rect(0, image.rows / 2, image.cols, image.rows / 2));
    cv::Mat hsv;
    cv::cvtColor(lowerHalf, hsv, cv::COLOR_BGR2HSV);
    cv::Scalar meanVal = cv::mean(hsv);
    // Assuming green is emphasized in the HSV's S channel
    return meanVal[1]; // Assuming green is emphasized in the HSV's S channel
}

std::pair<cv::Mat, cv::Mat> computeSpatialHistograms_texture(const cv::Mat &image, int bins)
{

    cv::Mat topHist = computeRGBHistogram(image, bins);
    cv::Mat bottomHist = texture(image, bins);
    // std::cout<<bottomHist<<std::endl;
    return {topHist, bottomHist};
}

cv::Mat texture(cv::Mat image, int bins)
{
    cv::Mat sobelx, sobely, grad, histogram, dst_img, grayscale;
    // cv::Mat feature = Mat::zeros(2, histSize, CV_32F);
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
    // sobelX3x3(grayscale,sobelx);
    // sobelY3x3(grayscale,sobely);
    // magnitude(sobelx,sobely,grad);
    cv::Sobel(grayscale, sobelx, CV_32F, 1, 0, 3);
    cv::Sobel(grayscale, sobely, CV_32F, 0, 1, 3);
    cv::magnitude(sobelx, sobely, grad);
    dst_img = orientation(grayscale, sobelx, sobely);

    histogram = computeRGChromaticityHistogram(dst_img, bins);

    // cv::Mat feature(2, histSize, CV_32F, cv::Scalar(0));
    cv::Mat feature = cv::Mat::zeros(bins, bins, CV_32F);

    // // calculate the range in each bin
    // float rangeMagnitude = 400 / 8.0;
    // float rangeOrientation = 2 * CV_PI / 8.0;

    // // loop the magnitude and orientation and build the 2D histogram
    // for (int i = 0; i < grad.rows; i++) {
    //     for (int j = 0; j < grad.cols; j++) {
    //         int m = grad.at<float>(i, j) / rangeMagnitude;
    //         int o = (dst_img.at<float>(i, j) + CV_PI) / rangeOrientation;
    //         std::cout << "m: " << m << " o: " << o << std::endl;
    //         feature.at<float>(m, o)++;
    //     }
    // }

    // L2 normalize the histogram
    // normalize(feature, feature, 1, 0, cv::NORM_L2, -1, cv::Mat());

    // convert the 2D histogram into a 1D vector

    return histogram;
}

cv::Mat gaborTexture(const cv::Mat &image, int bins)
{
    std::vector<float> feature;

    // convert image to grayscale
    cv::Mat grayscale, gaborKernel, filteredImage, histogram;
    cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
    int kernelSize = 31;      // Size of the Gabor kernel
    double sigma = 5;         // Standard deviation of the Gaussian envelope
    double theta = CV_PI / 4; // Orientation of the Gabor filter (in radians)
    double lambda = 10;       // Wavelength of the sinusoidal factor
    double gamma = 0.5;
    gaborKernel = cv::getGaborKernel(cv::Size(kernelSize, kernelSize), sigma, theta, lambda, gamma, 0, CV_32F);
    cv::filter2D(image, filteredImage, CV_32F, gaborKernel);
    // cv::normalize(filteredImage, filteredImage, 0, 255, cv::NORM_L2, CV_32F);
    histogram = computeRGChromaticityHistogram(filteredImage, bins);

    // get gabor kernels and apply to the grayscale image
    // float sigmaValue[] = {1.0, 2.0, 4.0};
    // for (auto s : sigmaValue) {
    //     for (int k = 0; k < 16; k++) {
    //         float t = k * CV_PI / 8;
    //         cv::Mat gaborKernel = cv::getGaborKernel( cv::Size(31,31), s, t, 10.0, 0.5, 0, CV_32F );
    //         cv::Mat filteredImage;
    //         std::vector<float> hist(9, 0);
    //         cv::filter2D(grayscale, filteredImage, CV_32F, gaborKernel);

    //         // calculate the mean and standard deviation of each filtered image
    //         cv::Scalar mean, stddev;
    //         meanStdDev(filteredImage, mean, stddev);
    //         feature.push_back(mean[0]);
    //         feature.push_back(stddev[0]);
    //     }
    // }

    // // L2 normalize the feature vector
    // normalize(feature, feature, 1, 0, cv::NORM_L2, -1, cv::Mat());

    return histogram;
}

std::pair<cv::Mat, cv::Mat> computeSpatialHistograms_gabor(const cv::Mat &image, int bins)
{

    cv::Mat topHist = computeRGBHistogram(image, bins);
    cv::Mat bottomHist = gaborTexture(image, bins);
    // std::vector<float> gaborResult = gaborTexture(image);

    // // Convert the vector to a single-row cv::Mat
    // cv::Mat bottomHist(1, gaborResult.size(), CV_32F);
    // for (int i = 0; i < gaborResult.size(); ++i) {
    //     bottomHist.at<float>(0, i) = gaborResult[i];
    // }

    return {topHist, bottomHist};
}

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
    // cv::normalize(dst, dst, 0, 255, cv::NORM_L1);

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
    // cv::normalize(dst, dst, 0, 255, cv::NORM_L1);

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
            double magnitude = sqrt(sobelXrowptr[j][0] * sobelXrowptr[j][0] + sobelYrowptr[j][0] * sobelYrowptr[j][0]);
            magnitude = std::min(255.0, std::max(0.0, magnitude)); // Ensure magnitude is within [0, 255]
            // loop over color channels
            for (int c = 0; c < 3; c++)
            {

                // std::cout<< "Magnitude: "<<magnitude<<std::endl;
                dptr[j][c] = static_cast<uchar>(magnitude);
            }
        }

        // cv::normalize(dst, dst, 0, 255, cv::NORM_L1);
    }

    return 0;
}

cv::Mat orientation(cv::Mat &image, cv::Mat sx, cv::Mat sy)
{
    // calculate sobelX and sobelY
    // cv::Mat sx = sobelX(image);
    // cv::Mat sy = sobelY(image);

    // calculate orientation
    cv::Mat dst(image.size(), CV_32F);
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            dst.at<float>(i, j) = atan2(sy.at<float>(i, j), sx.at<float>(i, j));
        }
    }

    return dst;
}