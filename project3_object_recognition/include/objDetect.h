/**
 * author: Harshit Kumar, Khushi Neema
 * date: Feb 19, 2024
 * purpose: Implements various object detection algorithms and functions
 * 1. Preprocess and threshold the video frame
 * 2. Cleans noise/holes from the image
 * 3. Assigns labels to the binary image by utilizing two-pass method
 * 4. Compute features of the connected components
 * 5. Load the feature database from a file
 * 6. Calculate the standard deviation of the features in the database
 * 7. Calculate the Euclidean distance between two feature vectors
 * 8. Classify an unknown object by comparing its feature vector to those in the object database
 * 9. Update the confusion matrix
 * 10. Make confusion matrix NxN square matrix
 * 11. Get the embedding of the input image using a pre-trained DNN
 * 12. Detect objects in the input image using MobileNet-SSD
 *
 */

#ifndef OBJ_DETECT_H
#define OBJ_DETECT_H

#include <opencv2/opencv.hpp>

/**
 * @brief Struct to store the features of an object
*/
struct ObjectFeatures
{
    double percentFilled; // Percentage of the bounding box filled by the object
    double aspectRatio;   // Aspect ratio of the bounding box
    double huMoments[7];  // Hu moments
    cv::Mat dnnEmbedding; // DNN embedding vector
};

/**
 * @brief Apply 5x5 Gaussian blur to the input image
 *
 * @param src input image
 * @param dst output image
 * @return int
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Calculate the dynamic threshold using k-means clustering algorithm
 *
 * @param src input image
 * @param k number of clusters
 * @return double
 */
double calculateDynamicThreshold(const cv::Mat &src, int k);

/**
 * @brief Apply custom thresholding to the input image
 *
 * @param grayImage input image
 * @param thresh threshold value
 * @param maxValue maximum value
 * @return cv::Mat
 */
cv::Mat customThreshold(const cv::Mat &grayImage, double thresh, double maxValue);

/**
 * @brief Preprocess and threshold the video frame
 *
 * @param frame input frame
 * @return cv::Mat
 */
cv::Mat preprocessAndThreshold(const cv::Mat &frame);
/**
 * @brief Cleans noise/holes from the image
 *
 *
 *@param src Input image frame
 *@param dst Applied cleaned filtered image
 *@param operation deciding what operation is needed to be performed (growing, shrinking, growing + shrinking, shrinking + growing)
 */
void morphologyEx(const cv::Mat &src, cv::Mat &dst, int operation, const cv::Mat &kernel);

/**
 * @brief Assigns labels to the binary image by utilizing DFS method
 *
 *
 *@param binaryImage Input image frame
 *@param labels storing labels for each pixel
 *@param i x-coordinae of the image
 *@param j y-coordinae of the image
 *@param label current label to be assigned
 */
// void dfs(const cv::Mat& binaryImage, cv::Mat& labels, int i, int j, int label);
/**
 * @brief Assigns labels to the binary image
 *
 *
 *@param binaryImage Input image frame
 *@param labeledImage getting labels for every pixels of an image
 */
// void connectedComponents(const cv::Mat& binaryImage, cv::Mat& labeledImage);

/**
 * @brief Assigns labels to the binary image by utilizing two-pass method
 *
 * @param binaryImage Input image frame
 * @param labeledImage getting labels for every pixels of an image
 */
void connectedComponentsTwoPass(const cv::Mat &binaryImage, cv::Mat &labeledImage);

/**
 * @brief Compute features of the connected components
 *
 * @param labeledImage labeled image
 * @param outputImage output image
 */
std::map<int, ObjectFeatures> computeFeatures(const cv::Mat &labeledImage, cv::Mat &outputImage);

/**
 * @brief Load the feature database from a file
 *
 * @param filename name of the file
 * @param embeddingType type of embedding: default or dnn
 * @return std::map<std::string, ObjectFeatures> map of object features
 */
std::map<std::string, ObjectFeatures> loadFeatureDatabase(const std::string &filename, const std::string &embeddingType = "default");

/**
 * @brief Calculate the standard deviation of the features in the database
 *
 * @param database map of object features
 * @return ObjectFeatures standard deviation of the features
 */
ObjectFeatures calculateStdDev(const std::map<std::string, ObjectFeatures> &database);

/**
 * @brief Calculate the Euclidean distance between two feature vectors
 *
 * @param f1 feature vector 1
 * @param f2 feature vector 2
 * @param stdev standard deviation of the features
 * @return double Euclidean distance
 */
double scaledEuclideanDistance(const ObjectFeatures &f1, const ObjectFeatures &f2, const ObjectFeatures &stdev);

/**
 * @brief Classify an unknown object by comparing its feature vector to those in the object database
 *
 * @param unknownObjectFeatures feature vector of the unknown object
 * @param database map of object features
 * @param stdev standard deviation of the features
 * @param minDistance minimum distance
 * @param embeddingType type of embedding: default or dnn
 * @return std::string label of the best match
 */
std::string classifyObject(const ObjectFeatures &unknownObjectFeatures, const std::map<std::string, ObjectFeatures> &database, const ObjectFeatures &stdev,
                           double minDistance = std::numeric_limits<double>::max(), std::string embeddingType = "default");

/**
 * @brief Update the confusion matrix
 *
 * @param matrix confusion matrix
 * @param trueLabel true label
 * @param classifiedLabel classified label
 * @return void
 */
void updateConfusionMatrix(std::map<std::string, std::map<std::string, int>> &matrix, const std::string &trueLabel, const std::string &classifiedLabel);

/**
 * @brief Make confusion matrix NxN square matrix
 *
 * @param matrix confusion matrix
 * @return void
 */
void makeMatrixNxN(std::map<std::string, std::map<std::string, int>> &matrix);

/**
 * @brief Get the embedding of the input image using a pre-trained DNN
 * @param cv::Mat src        thresholded and cleaned up image in 8UC1 format
 * @param cv::Mat ebmedding  holds the embedding vector after the function returns
 * @param cv::Rect bbox      the axis-oriented bounding box around the region to be identified
 * @param cv::dnn::Net net   the pre-trained network
 * @param int debug          1: show the image given to the network and print the embedding, 0: don't show extra info
 */
int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::Rect &bbox, cv::dnn::Net &net, int debug);

/**
 * @brief Detect objects in the input image using MobileNet-SSD
 * @param img input image
 * @param prototxt_path path to the prototxt file
 * @param model_path path to the model file
 * @return cv::Mat image with overlayed detected objects
 */
cv::Mat objectDetMobileNetSSD(cv::Mat img, std::string prototxt_path, std::string model_path);
#endif // OBJ_DETECT_H
