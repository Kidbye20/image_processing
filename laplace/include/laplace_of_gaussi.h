
#ifndef GUIDED_FILTER_LAPLACE_OF_GAUSSI_H
#define GUIDED_FILTER_LAPLACE_OF_GAUSSI_H


// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


cv::Mat laplace_extract_edges(const cv::Mat&);


cv::Mat laplace_of_gaussi(const cv::Mat& source, const int radius=2, const double sigma=0.7);


cv::Mat difference_of_gaussi(const cv::Mat& source, const int radius=2, const double sigma=.7, const double k=1.6);


#endif //GUIDED_FILTER_LAPLACE_OF_GAUSSI_H
