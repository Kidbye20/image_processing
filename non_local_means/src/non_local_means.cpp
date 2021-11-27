// C++
#include <cmath>
#include <iostream>
// self
#include "non_local_means.h"

namespace {
    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }
}

// 没错, 这里的那个 uchar 会溢出, 有点东西啊, 恶心
cv::Mat non_local_means_gray(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma) {
    // 先做一个计算领域相似性的权重模板, 先来最简单的均值模板
    const int window_len = (radius << 1) + 1;
    const int window_size = window_len * window_len;
    std::vector<double> weights_kernel(window_size, 1. / (window_size));
    const double sigma_2_inv = 1. / (sigma * sigma);
    // 收集目标图像的信息
    cv::Mat denoised = noise_image.clone();
    const int H = noise_image.rows;
    const int W = noise_image.cols;
    // 将图像 padding 一下
    const auto padded_image = make_pad(noise_image, radius, radius);
    const int H2 = padded_image.rows;
    const int W2 = padded_image.cols;
    const uchar* const padded_ptr = padded_image.data;
    // 现在开始滤波, 求目标图像中的每一点
    for(int i = 0;i < H; ++i) {
        const int left = std::max(radius, i - search_radius);
        const int right = std::min(W2, i + search_radius);
        for(int j = 0;j < W; ++j) {
            // 当前要去噪的点 (x, y), 以它为中心的区域的点, 我得收集起来
            uchar source[window_size];
            for(int t = 0;t < window_len; ++t)
                std::memcpy(source + t * window_len, padded_ptr + i * W2 + j, window_len * sizeof(uchar));
            // 累计值 和 权重总和
            double sum_value = 0;
            double weight_sum = 0;
            double weight_max = -1e3;
            // 每个点先确认它目前的搜索区域有多大
            const int up = std::max(radius, j - search_radius);
            const int down = std::min(H2, j + search_radius);
            // 在这个搜索区域搜索
            for(int x = left; x < right; ++x) {
                for(int y = up; y < down; ++y) {
                    // 如果碰到自己了, 不计算
                    if(x == i and y == j)
                        continue;
                    // 当前对比的区域是以 x, y 为中心, 半径为 radius 的区域
                    // 我得把这个区域的值都找出来, 收集起来
                    uchar target[window_size];
                    for(int t = 0;t < window_len; ++t)
                        std::memcpy(target + t * window_len, padded_ptr + x * W2 + y, window_len * sizeof(uchar));
                    // 然后计算两个区域的相似度
                    double distance = 0.0;
                    for(int k = 0;k < window_size; ++k) {
                        double res = static_cast<double>(target[k] - source[k]);
                        distance += weights_kernel[k] * (res * res);
                    }
                    const double cur_weight = std::exp(-distance * sigma_2_inv);
                    if(cur_weight > weight_max) weight_max = cur_weight;
                    sum_value += cur_weight * padded_image.at<uchar>(x, y);
                    weight_sum += cur_weight;
                }
            }
            // 搜索结束
            sum_value += weight_max * noise_image.at<uchar>(i, j);
            weight_sum += weight_max;
            denoised.at<uchar>(i, j) = cv::saturate_cast<uchar>(sum_value / weight_sum);
        }
    }
    return denoised;
}

// 搜索窗口大小 11x11, 邻域 5x5
cv::Mat non_local_means(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma) {
    const int C = noise_image.channels();
    if(C == 1) return non_local_means_gray(noise_image, search_radius, radius, sigma);
    return noise_image;
}
