#ifndef GAUSSI_FILTER_GUIDED_FILTER_H
#define GAUSSI_FILTER_GUIDED_FILTER_H

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


// 单通道(灰度图)的引导滤波去噪
cv::Mat guided_filter_channel(const cv::Mat& noise_image, const cv::Mat& guided_image, const int radius=2, const double epsilon=0.01);

// 对图像做 padding, 速度也更快
cv::Mat guided_filter_channel_padded(const cv::Mat& noise_image, const cv::Mat& guided_image, const int radius_h=2, const int radius_w=2, const double epsilon=0.01);

#endif //GAUSSI_FILTER_GUIDED_FILTER_H
