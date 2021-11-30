// C++
#include <cmath>
#include <vector>
#include <iostream>
// self
#include "non_local_means.h"


namespace {
    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }
    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}


namespace {
    // 这是最简单的积分图求法
    void compute_integral_image(const float* const source, float* const target, const int H, const int W) {
        target[0] = source[0];
        // 第一行
        for(int j = 1;j < W; ++j) target[j] = target[j - 1] + source[j];
        // 第一列
        for(int i = 1;i < H; ++i) target[i * W] = target[(i - 1) * W] + source[i * W];
        // 中间的部分
        for(int i = 1;i < H; ++i) {
            const float* const row_ptr = source + i * W;
            float* const res_ptr = target + i * W;
            const float* const old_res_ptr = target + (i - 1) * W;
            for(int j = 1;j < W; ++j)
                // 画图就知道了
                res_ptr[j] = row_ptr[j] + res_ptr[j - 1] + old_res_ptr[j] - old_res_ptr[j - 1];
        }
    }

    // 改进的 积分图求法
    void fast_compute_integral_image(const float* const source, float* const target, const int H, const int W) {
        // 首先第一行
        target[0] = source[0];
        for(int j = 1;j < W; ++j) target[j] = target[j - 1] + source[j];
        // 从第二行开始累加
        for(int i = 1;i < H; ++i) {
            // 这一行的临时变量
            float temp = 0.0;
            const float* src_ptr = source + i * W;
            float* row_ptr = target + i * W;
            float* old_row_ptr = target + (i - 1) * W;
            // 这一行到 j 的累加值, + 上一行同一列的值
            for(int j = 0;j < W; ++j) {
                temp += src_ptr[j];
                row_ptr[j] = temp + old_row_ptr[j];
            }
        }
    }
}

#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
void faster_compute_integral_image(const float* const source, float* const target, const int H, const int W) {
    // 首先第一行
    target[0] = source[0];
    for(int j = 1;j < W; ++j) target[j] = target[j - 1] + source[j];
    // 从第二行开始累加
    for(int i = 1;i < H; ++i) {
        // 这一行的临时变量
        float temp = 0.0;
        const float* src_ptr = source + i * W;
        float* row_ptr = target + i * W;
        float* old_row_ptr = target + (i - 1) * W;
        // 这一行到 j 的累加值, + 上一行同一列的值
        int pos = 0;
        for(; pos < W - 4; pos += 4) {
            // b, c, d 累加
            const float a = src_ptr[pos];
            const float b = a + src_ptr[pos + 1];
            const float c = b + src_ptr[pos + 2];
            const float d = c + src_ptr[pos + 3];
            // 上一行的值 + 这一行的累计值, 一次性加四个, 有点类似 cuda 的核
            __m128 result = _mm_add_ps(_mm_set_ps(a + temp, b + temp, c + temp, d + temp), _mm_set_ps(old_row_ptr[pos], old_row_ptr[pos + 1], old_row_ptr[pos + 2], old_row_ptr[pos + 3]));
            _mm_storeu_ps(&row_ptr[pos], result);
            temp += d;
        }
        // 剩余的没有凑满 4 的点
        for(;pos < W; ++pos) {
            temp += src_ptr[pos];
            row_ptr[pos] = old_row_ptr[pos] + temp;
        }
    }
}


// 参考 https://www.ipol.im/pub/art/2014/120/
// 参考 https://blog.csdn.net/haronchou/article/details/109223032
cv::Mat fast_non_local_means_gray_3(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma, const bool use_fast_exp) {
    // 获取图像信息
    const int H = noise_image.rows;
    const int W = noise_image.cols;
    const int length = H * W;
    // 对图像做补齐操作
    const int relative_pos = search_radius + radius;
    const auto padded_image = make_pad(noise_image, relative_pos, relative_pos);
    const int W2 = padded_image.cols;
    // 存储每个点的当前求和, 以及当前权重之和
    std::vector<float> cur_sum(length, 0);
    std::vector<float> weight_sum(length, 0);
    // 因为积分图, 计算时, 需要以每个点为中心的 kernel 的四个顶角的值, 所以 mse 均值图跟 mse 图都需要这四个顶角,
    // 所以当前相对位置跟当前图像的 "变大" 图像都要重新处理
    const int add_size = 2 * radius;
    const int H3 = H + add_size;
    const int W3 = W + add_size;
    const int length3 = H3 * W3;
    // 当前相对位置对应的图像
    cv::Mat relative_image = cv::Mat::zeros(H3, W3, CV_8UC1);
    // 当前图像, 变大的图像
    cv::Mat cur_image = cv::Mat::zeros(H3, W3, CV_8UC1);
    for(int t = 0;t < H3; ++t)
        std::memcpy(cur_image.data + t * W3, padded_image.data + (search_radius + t) * W2 + search_radius, W3 * sizeof(uchar));
    // 当前相对位置对应 MSE, 和平均 MSE, 注意不是 length, 而是 length3
    std::vector<float> relative_errors(length3, 0);
    std::vector<float> relative_mean(length3, 0);
    // 一些常量
    const float sigma_inv = 1. / (sigma * sigma);
    const float area_inv = 1. / ((add_size + 1) * (add_size + 1));
    // 遍历每一个相对位置
    for(int x = -search_radius; x <= search_radius; ++x) {
        for(int y = -search_radius; y <= search_radius; ++y) {
            // 首先把当前相对位置对应的图像抠出来
            for(int t = 0;t < H3; ++t)
                std::memcpy(relative_image.data + t * W3, padded_image.data + (search_radius + x + t) * W2 + relative_pos + y, W3 * sizeof(uchar));
            // 二者计算 mse
            for(int i = 0;i < length3; ++i) {
                const float error = relative_image.data[i] - cur_image.data[i];
                relative_errors[i] = error * error;
            }
            // 求 mse 的积分图
            faster_compute_integral_image(relative_errors.data(), relative_mean.data(), H3, W3);
            // 积分图每个点都除以核的大小, 得到 mse 的平均图
            for(int i = 0;i < length3; ++i) relative_mean[i] *= area_inv;
            // 遍历图像中每个点, 记录当前相对位置对这个点的权重
            for(int i = 0;i < H; ++i) {
                // 目标图像(denoised) 第 i 行的指针
                float* cur_sum_ptr = cur_sum.data() + i * W;
                float* weight_sum_ptr = weight_sum.data() + i * W;
                // 上面两个顶角所在那一行的指针
                float* mean_up_ptr = relative_mean.data() + i * W3;
                // 下面两个顶角所在那一行的指针
                float* mean_down_ptr = mean_up_ptr + add_size * W3;
                // 当前相对位置对应的像素, 第 i 行的指针
                uchar* const relative_row_ptr = relative_image.data + (radius + i) * W3 + radius;
                for(int j = 0;j < W; ++j) {
                    // (a, b) + (a - 1, b - 1) - (a - 1, b) - (a, b - 1)
                    const float distance = mean_down_ptr[j + add_size] + mean_up_ptr[j] - mean_down_ptr[j] - mean_up_ptr[j + add_size];
                    const float w = std::exp(-distance * sigma_inv);
                    cur_sum_ptr[j] += w * relative_row_ptr[j];
                    weight_sum_ptr[j] += w;
                }
            }
        }
    }
    auto denoised = noise_image.clone();
    for(int i = 0;i < length; ++i)
        denoised.data[i] = cv::saturate_cast<uchar>(cur_sum[i] / weight_sum[i]);
    return denoised;
}
