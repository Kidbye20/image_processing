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
    void box_filter(const double* const new_source, double* const sum_ptr, const int radius_h, const int radius_w, const int H, const int W) {
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
	}
}



cv::Mat fast_non_local_means_gray(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma) {
    // 收集图像信息
    int H = noise_image.rows;
    int W = noise_image.cols;
    const int length = H * W;
    // 把图像补齐
    const cv::Mat padded_image = make_pad(noise_image, search_radius + radius, search_radius + radius);
    const int W2 = padded_image.cols;
    // 临时存储每一个相对位置的当前加权和和当前权重和
    std::vector<double> cur_sum(length, 0);
    std::vector<double> weight_sum(length, 0);

    const double sigma_inv = 1. / (sigma * sigma);

    const int relative_pos = search_radius + radius;

    cv::Mat relative_image = noise_image.clone();

    std::vector<double> residual_image(length, 0);
    std::vector<double> residual_mean(length, 0);

    for (int x = -search_radius; x <= search_radius; ++x) {
        for (int y = -search_radius; y <= search_radius; ++y)  {
            // 开始拷贝
            for(int t = 0;t < H; ++t)
                // 这里很耗时间
                std::memcpy(relative_image.data + t * W, padded_image.data + (relative_pos + t + x) * W2 + relative_pos + y, sizeof(uchar) * W);
            // dfgcv_show(relative_image);
            for(int i = 0;i < length; ++i) {
                const double temp = noise_image.data[i] - relative_image.data[i];
                residual_image[i] = temp * temp;
            }
            box_filter(residual_image.data(), residual_mean.data(), radius, radius, H, W);
            for(int i = 0;i < length; ++i) {
                double distance = - residual_mean[i] * sigma_inv;
                double w = std::exp(distance);
                weight_sum[i] += w;
                cur_sum[i] += w * relative_image.data[i];
            }
        }
    }
    auto denoised = noise_image.clone();
    for(int i = 0;i < length; ++i)
        denoised.data[i] = cv::saturate_cast<uchar>(cur_sum[i] / weight_sum[i]);
    return denoised;
}