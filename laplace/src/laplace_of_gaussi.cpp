// C++
#include <cmath>
#include <iostream>
// self
#include "laplace_of_gaussi.h"


namespace {
    cv::Mat make_pad(const cv::Mat &one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }

    inline double fast_exp(const double y) {
        double d;
        *(reinterpret_cast<int*>(&d) + 0) = 0;
        *(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * y + 1072632447);
        return d;
    }
}


cv::Mat laplace_extract_edges(const cv::Mat& source) {
    // 获取信息
    const int H = source.rows;
    const int W = source.cols;
    // padding
    const auto padded_image = make_pad(source, 1, 1);
    const int W2 = padded_image.cols;
    // 准备结果
    auto result = source.clone();
    // 开始滤波
    for(int i = 0;i < H; ++i) {
        const uchar* row_ptr = padded_image.data + (1 + i) * W2 + 1;
        uchar* const res_ptr = result.data + i * W;
        for(int j = 0;j < W; ++j) {
            // 每个点, 找周围几个点, 算一下
            const uchar u = row_ptr[j - W2], d = row_ptr[j + W2], l = row_ptr[j - 1], r = row_ptr[j + 1];
            res_ptr[j] = cv::saturate_cast<uchar>(std::abs(u + d + l + r - 4 * row_ptr[j]));
//            const uchar u = row_ptr[j - W2], d = row_ptr[j + W2], l = row_ptr[j - 1], r = row_ptr[j + 1];
//            const uchar u_1 = row_ptr[j - W2], u_2 = row_ptr[j - 1], d_1 = row_ptr[j + W2], d_2 = row_ptr[j + 1];
//            double value = u + d + l + r + u_1 + u_2 + d_1 + d_2 - 8 * row_ptr[j];
//            if(value < 0) value = -value;
//            res_ptr[j] = cv::saturate_cast<uchar>(value);
        }
    }
    return result;
}




cv::Mat laplace_of_gaussi(const cv::Mat& source, const int radius, const double sigma) {
    // padding 处理边缘
    const auto padded_image = make_pad(source, radius, radius);
    const int W2 = padded_image.cols;
    // 准备一个 LOG 模板
    const int window_len = (radius << 1) + 1;
    const int window_size = window_len * window_len;
    const double sigma_2 = sigma * sigma;
    const double sigma_6 = sigma_2 * sigma_2 * sigma_2;
    double LOG[window_size];
    int LOG_offset[window_size];
    int offset = 0;
    for(int i = -radius; i <= radius; ++i) {
        for(int j = -radius; j <= radius; ++j) {
            const double distance = i * i + j * j;
            LOG[offset] = (distance - 2 * sigma_2) / sigma_6 * std::exp(-distance / (2 * sigma_2));
            LOG_offset[offset] = i * W2 + j;
            ++offset;
        }
    }
    // 准备结果
    auto result_image = source.clone();
    // 收集原始图像信息
    const int H = source.rows;
    const int W = source.cols;
    // LOG 模板扫过
    for(int i = 0;i < H; ++i) {
        const uchar* const row_ptr = padded_image.data + i * W2;
        uchar* const res_ptr = result_image.data + i * W;
        for(int j = 0;j < W; ++j) {
            // 开始卷积
            double conv_sum = 0;
            for(int k = 0;k < offset; ++k)
                conv_sum += LOG[k] * row_ptr[j + LOG_offset[k]];
            res_ptr[j] = cv::saturate_cast<uchar>(conv_sum);
        }
    }
    cv::threshold(result_image, result_image, 10, 255, cv::THRESH_BINARY);
    return result_image;
}





#include "faster_gaussi_filter.h"
cv::Mat difference_of_gaussi(const cv::Mat& source, const int radius, const double sigma, const double k) {
    // 两个高斯卷积
    const auto lhs = faster_2_gaussi_filter_channel(source, 2 * radius + 1, k * sigma, k * sigma);
    const auto rhs = faster_2_gaussi_filter_channel(source, 2 * radius + 1, sigma, sigma);
    // 准备结果
    cv::Mat result = source.clone();
    const int length = result.rows * result.cols;
    for(int i = 0;i < length; ++i) result.data[i] = cv::saturate_cast<uchar>(lhs.data[i] - rhs.data[i]);  // / (k - 1)
    // 大于 0 的全部置为 255
    cv::threshold(result, result, 1, 255, cv::THRESH_BINARY);
    return result;
}