/*Kush Suryavanshi
* CS5330 Spring2023
* Project 1
* Filters
*/

#include<cstdio>
#include<opencv2/opencv.hpp>
#include"filter.h"


using namespace cv;
using namespace std;

int greyscale(Mat& src, Mat& dst) {
	// Convert image to greyscale by modifying pixel values
	dst = Mat::zeros(src.size(), src.type());

	//loop over src 
	for (int i = 1; i < src.rows; i++) { // loop over row 
		// src pointer for row
		cv::Vec3b* rptr = src.ptr<cv::Vec3b>(i);
		// destination pointer
		cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i); 

		// for each column
		for (int j = 1;j < src.cols - 1;j++) {			
			// using cvtcolorfunction COLOR_BGR2GRAY equation to find grayscale value
			// RGB[A] to Gray:Y←0.299⋅R+0.587⋅G+0.114⋅B
			dptr[j][0] = 0.114 * rptr[j][0] + 0.587 * rptr[j][1] + 0.229 * rptr[j][2] ;
			dptr[j][1] = 0.114 * rptr[j][0] + 0.587 * rptr[j][1] + 0.229 * rptr[j][2] ;
			dptr[j][2] = 0.114 * rptr[j][0] + 0.587 * rptr[j][1] + 0.229 * rptr[j][2] ;
		}
	}
	return 0;
}

int blur5x5(cv::Mat& src, cv::Mat& dst)
{	// apply gaussian filter as separable filters


	// allocate dst image
	dst = Mat::zeros(src.size(), src.type());
	
	// loop over columns
	for (int i = 2; i < src.rows - 2; i++) {

		//src row pointers
		Vec3b* rptrm2 = src.ptr<Vec3b>(i - 2);
		Vec3b* rptrm1 = src.ptr<Vec3b>(i - 1);
		Vec3b* rptr = src.ptr<Vec3b>(i);
		Vec3b* rptrp1 = src.ptr<Vec3b>(i + 1);
		Vec3b* rptrp2 = src.ptr<Vec3b>(i + 2);

		//destination ptr
		Vec3b* dptr = dst.ptr<Vec3b>(i);

		// loop over columns
		for (int j = 2; j < src.cols - 2; j++) {

			// loop over color channels
			for (int c = 0; c < 3; c++) {
				
				//row filter [1 , 2, 4, 2, 1]
				dptr[j][c] = 1 * rptr[j - 2][c] + 2 * rptr[j - 1][c] + 4 * rptr[j][c] + 2 * rptr[j + 1][c] + 1 * rptr[j + 2][c];

				/*column filter
					[1]  m2
					[2]  m1
					[4]  r
					[2]  p1
					[1]  p2
				*/

				dptr[j][c] = (dptr[j][c] + 1 * rptrm1[j][c] + 2 * rptrm1[j][c] + 4 * rptr[j][c] + 2 * rptrp1[j][c] + 1 * rptrp2[j][c]) / 10;
			}
		}
	}
	return 0;
}


int sobelX3x3(Mat& src, Mat& dst) {
	// apply 3x3 SobelX filter
	//[-1, 0, 1]
	//[-2, 0, 2]
	//[-1, 0, 1]
	// allocate dst image
	dst = Mat::zeros(src.size(), CV_16SC3);

	// loop over columns
	for (int i = 1; i < src.rows - 1; i++) {

		//src row pointers
		Vec3b* rptrm1 = src.ptr<Vec3b>(i - 1);
		Vec3b* rptr = src.ptr<Vec3b>(i);
		Vec3b* rptrp1 = src.ptr<Vec3b>(i + 1);

		//destination ptr
		Vec3s* dptr = dst.ptr<Vec3s>(i);

		// loop over columns
		for (int j = 1; j < src.cols - 1; j++) {

			// loop over color channels
			for (int c = 0; c < 3; c++) {
				

				dptr[j][c] = ( - 1 * rptrm1[j - 1][c] + 1 * rptrm1[j + 1][c] +
							 -2 * rptr[j - 1][c] + 2 * rptr[j + 1][c] +
							 -1 * rptrp1[j - 1][c] + 1 * rptrp1[j + 1][c])/4;
			}
		}
	}

	return 0;
}

int sobelY3x3(Mat& src, Mat& dst) {
	//apply 3x3 sobelY filter
	//[1, 2, 1]
	//[0, 0, 0]
	//[-1, -2, -1]
	
	dst = Mat::zeros(src.size(), CV_16SC3);

	// loop over columns
	for (int i = 1; i < src.rows - 1; i++) {

		//src row pointers
		Vec3b* rptrm1 = src.ptr<Vec3b>(i - 1);
		Vec3b* rptr = src.ptr<Vec3b>(i);
		Vec3b* rptrp1 = src.ptr<Vec3b>(i + 1);

		//destination ptr
		Vec3s* dptr = dst.ptr<Vec3s>(i);

		// loop over columns
		for (int j = 1; j < src.cols - 1; j++) {

			// loop over color channels
			for (int c = 0; c < 3; c++) {
				dptr[j][c] =( 1 * rptrm1[j - 1][c] + 2 * rptrm1[j][c] + 1 * rptrm1[j + 1][c] +
					         -1 * rptrp1[j - 1][c] + -2 * rptrp1[j][c] + -1 * rptrp1[j + 1][c])/4;
			}
		}
	}

	return 0;
}

int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst)
{	// Generate Gradient magnitude where  I = sqrt( sx*sx + sy*sy )

	dst = Mat::zeros(sx.size(), CV_8UC3);
	
	// loop over columns
	for (int i = 0; i < sx.rows; i++) {

		//src row pointers
		Vec3s* sxrptr = sx.ptr<Vec3s>(i);
		Vec3s* syrptr = sy.ptr<Vec3s>(i);

		//destination ptr
		Vec3b* dptr = dst.ptr<Vec3b>(i);

		// loop over columns
		for (int j =0 ; j < sx.cols; j++) {

			// loop over color channels
			for (int c = 0; c < 3; c++) {
				dptr[j][c] = sqrt((sxrptr[j][c] * sxrptr[j][c] + syrptr[j][c] * syrptr[j][c]));
			}
		}
	}

	return 0;
}

int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels)
{ // Task 8: apply blur quantize filter 
	
	dst = Mat::zeros(src.size(), src.type());
	
	Mat xt;
	xt = Mat::zeros(src.size(), src.type());

	int b;
	b = 255 / levels;
	
	Mat x;
	blur5x5(src, x);
	

	for (int i = 0; i < x.rows; i++) {
		Vec3b* xrptr = x.ptr<Vec3b>(i);
		Vec3b* xtrptr = xt.ptr<Vec3b>(i);
		Vec3b* drptr = dst.ptr<Vec3b>(i);
		for (int j = 0; j < x.cols; j++) {

			for (int c = 0; c < 3; c++) {
				xtrptr[j][c] = xrptr[j][c] / b;
				drptr[j][c] = xtrptr[j][c] * b;
			}
		}
	}

	return 0;
}

int cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold)
{	
	dst = Mat::zeros(src.size(), src.type());
	
	Mat sx;
	sobelX3x3(src, sx);
	Mat sy;
	sobelY3x3(src, sy);
	Mat gradient;
	magnitude(sx, sy, gradient);
	blurQuantize(src, dst, levels);
	
	for (int i = 0; i < gradient.rows; i++) {
		Vec3b* mrptr = gradient.ptr<Vec3b>(i);
		Vec3b* drptr = dst.ptr<Vec3b>(i);
		for (int j = 0; j < gradient.cols; j++) {

			if (mrptr[j][0] > magThreshold && mrptr[j][1] > magThreshold && mrptr[j][2] > magThreshold){
				for (int c = 0; c < 3; c++) {
					drptr[j][c] = 0;
				}
			}
			
		}
	}
	
	return 0;
}

int MeanBlur(cv::Mat& src, cv::Mat& dst)
{	//Applyting 5x5 mean Blur filter

	// allocate dst image
	dst = Mat::zeros(src.size(), src.type());

	// loop over columns
	for (int i = 2; i < src.rows - 2; i++) {

		//src row pointers
		Vec3b* rptrm2 = src.ptr<Vec3b>(i - 2);
		Vec3b* rptrm1 = src.ptr<Vec3b>(i - 1);
		Vec3b* rptr = src.ptr<Vec3b>(i);
		Vec3b* rptrp1 = src.ptr<Vec3b>(i + 1);
		Vec3b* rptrp2 = src.ptr<Vec3b>(i + 2);

		//destination ptr
		Vec3b* dptr = dst.ptr<Vec3b>(i);

		// loop over columns
		for (int j = 2; j < src.cols - 2; j++) {

			// loop over color channels
			for (int c = 0; c < 3; c++) {

				dptr[j][c] = (	rptrm2[j - 2][c] + rptrm2[j - 1][c] + rptrm2[j][c] + rptrm2[j + 1][c] + rptrm2[j + 2][c] +
								rptrm1[j - 2][c] + rptrm1[j - 1][c] + rptrm1[j][c] + rptrm1[j + 1][c] + rptrm1[j + 2][c] +
								rptr[j - 2][c]   + rptr[j - 1][c]   + rptr[j][c]   + rptr[j + 1][c]   + rptr[j + 2][c]   +
								rptrp1[j - 2][c] + rptrp1[j - 1][c] + rptrp1[j][c] + rptrp1[j + 1][c] + rptrp1[j + 2][c] +
								rptrp2[j - 2][c] + rptrp2[j - 1][c] + rptrp2[j][c] + rptrp2[j + 1][c] + rptrp2[j + 2][c]) / 25;
			}
		}
	}

	return 0;
}

int sepia(cv::Mat& src, cv::Mat& dst, int intensity, int blue, int green, int red)
{
	dst = Mat::zeros(src.size(), CV_16FC3);

	Mat kernel = (Mat_<float>(3,3)
					<<
					0.272, 0.534, 0.131,
					0.349, 0.686, 0.168,
					0.393, 0.769, 0.189);
	transform(src, dst, kernel);
	
	return 0;
}

