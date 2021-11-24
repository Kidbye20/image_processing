#ifndef GAUSSI_FILTER_GUIDED_FILTER_H
#define GAUSSI_FILTER_GUIDED_FILTER_H

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



cv::Mat guided_filter(const cv::Mat& noise_image, const cv::Mat& guided_image, const int radius=3, const double eta=0.01);

#endif //GAUSSI_FILTER_GUIDED_FILTER_H
