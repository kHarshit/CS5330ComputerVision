/**
 * author: Harshit Kumar
 * date: Jan 18, 2024
 * purpose: Display an image and exit when the user presses 'q'.
 * 
*/

#ifndef IMGDISPLAY_H
#define IMGDISPLAY_H

#include <opencv2/opencv.hpp>

/**
 * @brief Display an image and wait for the user to press 'q' to exit.
 * 
 * This program reads an image file, displays it in a window, and enters a loop
 * checking for a keypress. If the user types 'q', the program quits.
 * 
 * @param imagePath full path to the image file.
 * @return 0 if the operation is successful.
 */

int displayImage(const std::string& imgPath);

#endif // IMGDISPLAY_H
