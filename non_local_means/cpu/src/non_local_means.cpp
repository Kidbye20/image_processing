// C++
#include <cmath>
#include <cstring>
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


std::vector<double> get_kernel(const int window_size, const char* kernel_type) {
    // 权重模板是均值的话
    if(std::strcmp(kernel_type, "mean") == 0)
        return std::vector<double> (window_size, 1. / (window_size));
    // 高斯模板
    else if(std::strcmp(kernel_type, "gaussi") == 0) {
        std::vector<double> weight_kernel(window_size, 0);
        int offset = -1;
        double kernel_weight_sum = 0.0;
        const int radius = (int(std::sqrt(window_size)) - 1) >> 1;
        // 半径应该是 3 sigma 差不多了
        const double variance = int((2 * radius + 1) / 3);
        const double variance_2 = -0.5 / (variance * variance);
        for(int i = -radius; i <= radius; ++i)
            for(int j = -radius; j <= radius; ++j) {
                weight_kernel[++offset] = std::exp(variance_2 * (i * i + j * j));
                kernel_weight_sum += weight_kernel[offset];
            }
        for(int i = 0;i < window_size; ++i) weight_kernel[i] /= kernel_weight_sum;
        return weight_kernel;
    }
    // 没声明的话, 返回全 0 模板
    else return std::vector<double>(window_size, 0);

}

cv::Mat non_local_means_gray(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma, const char* kernel_type) {
    // 先做一个计算领域相似性的权重模板, 先来最简单的均值模板
    const int window_len = (radius << 1) + 1;
    const int window_size = window_len * window_len;
    const auto weights_kernel = get_kernel(window_size, kernel_type);
    const double sigma_2_inv = 1. / (sigma * sigma);
    // 收集目标图像的信息
    cv::Mat denoised = noise_image.clone();
    const int H = noise_image.rows;
    const int W = noise_image.cols;
    // 将图像 padding 一下
    const auto padded_image = make_pad(noise_image, radius, radius);
    const int H2 = padded_image.rows;
    const int W2 = padded_image.cols;
    const uchar* const noise_ptr = noise_image.data;
    const uchar* const padded_ptr = padded_image.data;
    // 现在开始滤波, 求目标图像中的每一点
    for(int i = 0;i < H; ++i) {
        const int left = std::max(radius, i - search_radius);
        const int right = std::min(W2, i + search_radius);
        for(int j = 0;j < W; ++j) {
            // 当前要去噪的点 (i, j), 以它为中心的区域的点, 我得收集起来
            uchar source[window_size];
            for(int t = 0;t < window_len; ++t)
                std::memcpy(source + t * window_len, padded_ptr + (i + t) * W2 + j, window_len * sizeof(uchar));
            // 累计值 和 权重总和
            double sum_value = 0;
            double weight_sum = 0;
            double weight_max = -1e3;
            // 每个点先确认它目前的搜索区域有多大, 为什么是 radius?
            const int up = std::max(radius, j - search_radius);
            const int down = std::min(H2, j + search_radius);
            // 在这个搜索区域搜索
            for(int x = left; x < right; ++x) {
                for(int y = up; y < down; ++y) {
                    // (i, j) 是相对于原图来说的位置, (x, y) 是相对于 padded 之后的图像来说的
                    // 如果碰到自己了, 不计算
                    if(x == i and y == j)
                        continue;
                    // 当前对比的区域是以 x, y 为中心, 半径为 radius 的区域
                    // 我得把这个区域的值都找出来, 收集起来
                    uchar target[window_size];
                    for(int t = 0;t < window_len; ++t)
                        std::memcpy(target + t * window_len, padded_ptr + (x - radius + t) * W2 + y - radius, window_len * sizeof(uchar));
                    // 然后计算两个区域的相似度
                    double distance = 0.0;
                    for(int k = 0;k < window_size; ++k) {
                        double res = static_cast<double>(target[k] - source[k]);
                        distance += weights_kernel[k] * (res * res);
                    }
                    const double cur_weight = std::exp(-distance * sigma_2_inv);
                    // 记录当前最大的权值
                    if(cur_weight > weight_max) weight_max = cur_weight;
                    // 累加值
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




namespace {
    // uint8 -> double, 生成
    std::vector<double> uchar2double(const cv::Mat& source, const int length) {
        std::vector<double> double_image(length);
        const uchar* const data_ptr = source.data;
        for(int i = 0;i < length; ++i) double_image[i] = (double)data_ptr[i] / 255;
        return double_image;
    }
    // uint8 -> double, 更新
    void uchar2double(const uchar* const src_ptr, double* const des_ptr, const int length) {
        for(int i = 0;i < length; ++i)
            des_ptr[i] = (double)src_ptr[i] / 255;
    }
    // 均值滤波
    std::vector<double> box_filter(const double* const new_source, const int radius_h, const int radius_w, const int H, const int W) {
        // 先对图像做 padding
        const int new_H = H + 2 * radius_h;
        const int new_W = W + 2 * radius_w;
        std::vector<double> padding_image(new_H * new_W, 0);
        double* const padding_ptr = padding_image.data();
        // 先把已有内容填上
        for(int i = 0;i < H; ++i) {
            double* const row_ptr = padding_ptr + (radius_h + i) * new_W + radius_w;
            const double* const src_row_ptr = new_source + i * W;
            std::memcpy(row_ptr, src_row_ptr, sizeof(double) * W);
        }
        // 填充上面的边界
        for(int i = 0;i < radius_h; ++i) {
            std::memcpy(padding_ptr + (radius_h - 1 - i) * new_W + radius_w, new_source + i * W, sizeof(double) * W);
            std::memcpy(padding_ptr + (new_H - radius_h + i) * new_W + radius_w, new_source + (H - i - 1) * W, sizeof(double) * W);
        }
        // 填充左右两边的边界, 这次没法 memcpy 了, 内存不是连续的
        for(int j = 0;j < radius_w; ++j) {
            double* const _beg = padding_ptr + radius_h * new_W + radius_w - 1 - j;
            for(int i = 0;i < H; ++i)
                _beg[i * new_W] = new_source[i * W + j];
        }
        for(int j = 0;j < radius_w; ++j) {
            double* const _beg = padding_ptr + radius_h * new_W + radius_w + W + j;
            for(int i = 0;i < H; ++i)
                _beg[i * new_W] = new_source[i * W + W - 1 - j];
        }
        // 现在图像的高和宽分别是 new_H, new_W, 草稿画一下图就知道
        const int kernel_h = (radius_h << 1) + 1;
        const int kernel_w = (radius_w << 1) + 1;
        // 准备 buffer 和每一个点代表的 box 之和
        std::vector<double> buffer(new_W, 0.0);
        std::vector<double> sum(H * W, 0.0);
        double* const sum_ptr = sum.data();
        // 首先求目标(结果的)第一行的 buffer
        for(int i = 0;i < kernel_h; ++i) {
            const double* const row_ptr = padding_ptr + i * new_W;
            for(int j = 0;j < new_W; ++j) buffer[j] += row_ptr[j];
        }
        // 求每一行的每个点的 box 的和
        for(int i = 0;i < H; ++i) {
            // 当前 kernel_w 个 buffer 点的累加值
            double cur_sum = 0;
            // 这一行第一个 box 的 cur_sum, 前 kernel_w 个 buffer 点的累加值
            for(int j = 0;j < kernel_w; ++j) cur_sum += buffer[j];
            // 记录这第一个 box 的值
            const int _beg = i * W;
            sum_ptr[_beg] = cur_sum;
            // 向右边挪动, 减去最左边的值, 加上最右边要加进来的值
            for(int j = 1;j < W; ++j) {
                cur_sum = cur_sum - buffer[j - 1] + buffer[j - 1 + kernel_w];
                sum_ptr[_beg + j] = cur_sum;
            }
            // 这一行的点的 sum 都记下来了, 准备换行, 更新 buffer ==> 减去最上面的值, 加上新一行对应的值
            // 最后一次不需要更新......
            if(i != H - 1) {
                const double* const up_ptr = padding_ptr + i * new_W;
                const double* const down_ptr = padding_ptr + (i + kernel_h) * new_W;
                for(int j = 0;j < new_W; ++j) buffer[j] = buffer[j] - up_ptr[j] + down_ptr[j];
            }
        }
        // sum 其实就是最后的矩阵, 现在要除以 area, 每个 box 的面积
        const int area = kernel_h * kernel_w;
        const int length = H * W;
        for(int i = 0;i < length; ++i)
            sum_ptr[i] /= area;
		return sum;
	}
}




cv::Mat fast_non_local_means_gray(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma, const char* kernel_type) {
    // 先做一个计算领域相似性的权重模板, 先来最简单的均值模板
    const int window_len = (radius << 1) + 1;
    const int window_size = window_len * window_len;
    const auto weights_kernel = get_kernel(window_size, kernel_type);
    const double sigma_2_inv = 1. / (sigma * sigma);
    // 收集目标图像的信息
    cv::Mat denoised = noise_image.clone();
    const int H = noise_image.rows;
    const int W = noise_image.cols;
    const int length = H * W;
    // 将图像 padding 一下, 这次的 padding 跟前面不一样
    const auto padded_image = make_pad(noise_image, radius + search_radius, radius + search_radius);
    const int H2 = padded_image.rows;
    const int W2 = padded_image.cols;
    // 准备几个中间变量
    std::vector<double> sum_value(length, 0.0);
    std::vector<double> weight_sum(length, 0.0);
    std::vector<double> weight_max(length, 1e-3);
    // 输入图的 double 图像
    const auto noise_double_image = uchar2double(noise_image, length);
    // 每次相对位置代表的那个图像
    std::vector<double> relative_double_image(length, 0.0);
    // 上面两张图象的差的平方
    std::vector<double> residual(length, 0.0);
    double* const residual_ptr = residual.data();
    // 从每一个相对位置开始计算
    const int relative = radius + search_radius;
    for(int x = -search_radius; x <= search_radius; ++x) {
        for(int y = -search_radius; y <= search_radius; ++y) {
            // 如果和当前点一模一样
            if(x == 0 and y == 0) continue;
            // 首先, 当前相对位置有一个图像, 先把它抠出来
            const auto relative_image = padded_image(cv::Rect(relative + x, relative + y, W, H));
            uchar2double(relative_image.data, relative_double_image.data(), length);
            // 接下来, 重头戏,
            // (relative_double_image - noise_double_image) ^ 2 的积分图
            for(int i = 0;i < length; ++i) {
                const double temp = relative_double_image[i] - noise_double_image[i];
                residual_ptr[i] = temp * temp / window_size; // 均值滤波
            }
            // 算 box_filter
            const auto residual_mean = box_filter(residual_ptr, radius, radius, H, W);
            // 现在开始图像中的每一个点, 开始累计
            for(int i = 0;i < length; ++i) {
                const double cur_weight = std::exp(-residual_mean[i] * sigma_2_inv);
                weight_sum[i] += cur_weight;
                sum_value[i] += cur_weight * noise_double_image[i];
                if(cur_weight > weight_max[i]) weight_max[i] = cur_weight;
            }
        }
    }
    // 结束之后
    std::cout << "length  " << length << std::endl;
    for(int i = 0;i < length; ++i) sum_value[i] += weight_max[i] * noise_double_image[i];
    for(int i = 0;i < length; ++i) weight_sum[i] += weight_max[i];
    for(int i = 0;i < length; ++i) denoised.data[i] = cv::saturate_cast<uchar>(255 * (sum_value[i] / weight_sum[i]));
    return denoised;
}


// 如果 fast? 用积分图和 box_filter 之类的?




// 有两种写法
cv::Mat non_local_means_color(const cv::Mat& noise_image, const int search_radius=5, const int radius=2, const int sigma=1, const char* kernel_type="mean") {
    return  noise_image;
}


// 搜索窗口大小 11x11, 邻域 5x5
cv::Mat non_local_means(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma, const char* kernel_type, const bool fast) {
    const int C = noise_image.channels();
    if(C == 1)
        if(fast == false) return non_local_means_gray(noise_image, search_radius, radius, sigma, kernel_type);
        else return fast_non_local_means_gray(noise_image, search_radius, radius, sigma, kernel_type);
    return noise_image;
}
