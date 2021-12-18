
#ifndef GUIDED_FILTER_LAPLACE_OF_GAUSSI_H
#define GUIDED_FILTER_LAPLACE_OF_GAUSSI_H


// C++
#include <list>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


cv::Mat laplace_extract_edges(const cv::Mat&);


cv::Mat laplace_of_gaussi_edge_detection(const cv::Mat& source, const int radius=2, const double sigma=0.7, const double threshold=100);


using keypoints_type = std::list<std::pair<double, double> >;

std::pair< keypoints_type, keypoints_type > laplace_of_gaussi_keypoints_detection(const cv::Mat& source, const int radius=2, const std::vector< std::pair<double, double > > sigma_list={{0.3, 0.4}, {0.6, 0.7}, {0.7, 0.8}}, const double threshold=100);

#endif //GUIDED_FILTER_LAPLACE_OF_GAUSSI_H

