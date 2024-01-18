/**
 * author: Harshit Kumar
 * date: Jan 18, 2024
 * purpose: Display video, allowing users to save frames and quit using keystrokes.
 * 
*/

#ifndef VIDDISPLAY_H
#define VIDDISPLAY_H

#include <opencv2/opencv.hpp>

/**
 * @brief Display video, allowing users to save frames and quit using keystrokes.
 * 
 * @param videoDeviceIndex Index of the video device (e.g., 0 for the default camera). 
 * @return 0 if the operation is successful.
 */

int displayVideo(int videoDeviceIndex=0);

#endif // VIDDISPLAY_H
