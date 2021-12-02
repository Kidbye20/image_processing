#ifndef GUIDED_FILTER_DARK_CHANNEL_PRIOR_H
#define GUIDED_FILTER_DARK_CHANNEL_PRIOR_H

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



std::map<const std::string, cv::Mat> dark_channel_prior_dehaze(const cv::Mat& haze_image, const int radius=3, const double top_percent=0.001, const double t0=0.1, const double omega=0.95, const bool guided=false);

#endif //GUIDED_FILTER_DARK_CHANNEL_PRIOR_H
