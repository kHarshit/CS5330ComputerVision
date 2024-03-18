/**
 * author: Harshit Kumar, Khushi Neema
 * date: Mar 5th, 2024
 * purpose: Finding and drawing chessboard corners
 *
 */

#ifndef CHESSBOARDCORNER_H
#define CHESSBOARDCORNER_H

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;



/**
 * @brief Draws the chessboard corners on the frame
 * @param frame input frame
 * @param boardSize size of the chessboard
 * @param corner_set list of all the corners in the chessboard
 * @return true if the corners are found, false otherwise
*/
bool drawchessboardcorner(cv::Mat frame, cv::Size boardSize, std::vector<cv::Point2f> &corner_set);

/**
 * @brief Saves the calibration points
 * @param corner_set list of all the corners in the chessboard
 * @param corner_list list of all the corners in the chessboard
 * @param point_set list of all the points in the chessboard
 * @param point_list list of all the points in the chessboard
 * @param flag flag to check if the points are saved
 * @param boardSize size of the chessboard
 * @return void
*/
void saveCalibrationPoints(vector<cv::Point2f>& corner_set, vector<vector<cv::Point2f>>& corner_list, vector<cv::Vec3f>& point_set, vector<vector<cv::Vec3f>>& point_list, int& flag, Size boardSize);

/**
 * @brief Calibrates the camera and saves the parameters
 * @param point_list list of all the points in the chessboard
 * @param corner_list list of all the corners in the chessboard
 * @param frame_size size of the frame
 * @param camera_matrix camera matrix
 * @param distortion_coefficients distortion coefficients
 * @return void
*/
void calibrateCameraAndSaveParameters(std::vector<std::vector<cv::Vec3f>>& point_list, std::vector<std::vector<cv::Point2f>>& corner_list, cv::Size frame_size, cv::Mat& camera_matrix, cv::Mat& distortion_coefficients);

/**
 * @brief Calculates the pose of the chessboard
 * @param corner_set list of all the corners in the chessboard
 * @param camera_matrix camera matrix
 * @param distortion_coefficients distortion coefficients
 * @param boardSize size of the chessboard
 * @param rvec rotation vector
 * @param tvec translation vector
 * @return void
*/
void calculatePose(const std::vector<cv::Point2f>& corner_set, const cv::Mat& camera_matrix, const cv::Mat& distortion_coefficients, const cv::Size& boardSize, cv::Mat& rvec, cv::Mat& tvec);

/**
 * @brief Projects the points and draws them on the frame
 * @param corner_set list of all the corners in the chessboard
 * @param rvec rotation vector
 * @param tvec translation vector
 * @param camera_matrix camera matrix
 * @param distortion_coefficients distortion coefficients
 * @param boardSize size of the chessboard
 * @param image input frame
 * @return void
*/
void projectPointsAndDraw(const std::vector<cv::Point2f>& corner_set, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& camera_matrix, const cv::Mat& distortion_coefficients, const cv::Size& boardSize, cv::Mat& image);


/**
 * @brief Creating virtual object in the frame
 * @param rvec rotation vector
 * @param tvec translation vector
 * @param camera_matrix camera matrix
 * @param distortion_coefficients distortion coefficients
 * @param boardSize size of the chessboard
 * @param image input frame
 * @return void
*/
void createObject(const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& camera_matrix, const cv::Mat& distortion_coefficients, const cv::Size& boardSize, cv::Mat& image);

/**
 * @brief Blurs the outside chessboard region
 * @param boardSize size of the chessboard
 * @param rvec rotation vector
 * @param tvec translation vector
 * @param camera_matrix camera matrix
 * @param distortion_coefficients distortion coefficients
 * @param image input frame
 * @return void
*/
void blurOutsideChessboardRegion(const cv::Size& boardSize, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& camera_matrix, const cv::Mat& distortion_coefficients, cv::Mat& image);

/**
 * @brief Blends the chessboard region
 * @param boardSize size of the chessboard
 * @param rvec rotation vector
 * @param tvec translation vector
 * @param camera_matrix camera matrix
 * @param distortion_coefficients distortion coefficients
 * @param image input frame
 * @param texture texture to be blended
 * @return void
*/
void blendChessboardRegion(const cv::Size& boardSize, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& camera_matrix, const cv::Mat& distortion_coefficients, cv::Mat& image, const cv::Mat& texture);

/**
 * @brief Blends the outside chessboard region
 * @param boardSize size of the chessboard
 * @param rvec rotation vector
 * @param tvec translation vector
 * @param camera_matrix camera matrix
 * @param distortion_coefficients distortion coefficients
 * @param image input frame
 * @param pebbles texture to be blended
 * @return void
*/
void blendOutsideChessboardRegion(const cv::Size& boardSize, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& camera_matrix, const cv::Mat& distortion_coefficients, cv::Mat& image, const cv::Mat& pebbles);
#endif