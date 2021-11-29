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



cv::Mat fast_non_local_means_gray(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma)
{
    cv::Mat padded_image = make_pad(noise_image, search_radius + radius, search_radius + radius);
    cv::Mat noise_double_image, padded_double_image;
    noise_image.convertTo(noise_double_image, CV_64FC1);
    padded_image.convertTo(padded_double_image, CV_64FC1);

    int H = noise_double_image.rows;
    int W = noise_double_image.cols;
    const int length = H * W;

    cv::Mat cur_sum(H, W, CV_64FC1, 0.0);
    cv::Mat weight_sum(H, W, CV_64FC1, 0.0);

    const double sigma_inv = 1. / (sigma * sigma);

    cv::Mat St(H, W, CV_64FC1, 0.0);

    for (int x = -search_radius; x <= search_radius; x++) {
        for (int y = -search_radius; y <= search_radius; y++)  {
            const auto ori_image = padded_double_image(cv::Rect(search_radius + radius + x, search_radius + radius + y, W, H));
            cv::Mat Disy = noise_double_image - ori_image;
            Disy = Disy.mul(Disy);

            cv::boxFilter(Disy, St, CV_64FC1, cv::Size(2 * radius + 1, 2 * radius + 1));

            for (int i = 0; i < H; i++) {
                double *weight_sum_p = weight_sum.ptr<double>(i);
                double *cur_sum_p = cur_sum.ptr<double>(i);
                for (int j = 0; j < W; j++) {
                    double Disy = - St.at<double>(i, j) * sigma_inv;
                    double w = std::exp(Disy);
                    weight_sum_p[j] += w;
                    cur_sum_p[j] += w * ori_image.at<double>(i, j);
                }
            }
        }
    }
    cur_sum = cur_sum / weight_sum;
    auto radiust = noise_image.clone();
    cur_sum.convertTo(radiust, CV_8U);
    return  radiust;
}





// Eigen3 矩阵优化



//cv::Mat fast_non_local_means_gray(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma, const char* kernel_type) {
//    // 先做一个计算领域相似性的权重模板, 先来最简单的均值模板
//    const int window_len = (radius << 1) + 1;
//    const int window_size = window_len * window_len;
//    const double sigma_2_inv = 1. / (sigma * sigma);
//    // 收集目标图像的信息
//    cv::Mat denoised = noise_image.clone();
//    const int H = noise_image.rows;
//    const int W = noise_image.cols;
//    const int length = H * W;
//    // 将图像 padding 一下, 这次的 padding 跟前面不一样
//    const auto padded_image = make_pad(noise_image, radius + search_radius, radius + search_radius);
//    const int H2 = padded_image.rows;
//    const int W2 = padded_image.cols;
//    // 准备几个中间变量
//    std::vector<double> sum_value(length, 0.0);
//    std::vector<double> weight_sum(length, 0.0);
//    std::vector<double> weight_max(length, 1e-3);
//    // 输入图的 double 图像
//    const auto noise_double_image = uchar2double(noise_image, length);
//    // 每次相对位置代表的那个图像
//    // 上面两张图象的差的平方
//    std::vector<double> residual(length, 0.0);
//    double* const residual_ptr = residual.data();
//    // 从每一个相对位置开始计算
//    const int relative = radius + search_radius;
//    for(int x = -search_radius; x <= search_radius; ++x) {
//        for(int y = -search_radius; y <= search_radius; ++y) {
//            // 如果和当前点一模一样
//            if(x == 0 and y == 0) continue;
//            // 首先, 当前相对位置有一个图像, 先把它抠出来
//            const auto relative_image = padded_image(cv::Rect(relative + x, relative + y, W, H));
//            for(int i = 0;i < 10; ++i)
//                std::cout << double(relative_image.data[i]) << " ";
//            std::cout << std::endl;
//            const auto relative_double_image = uchar2double(relative_image, length);
//            // 接下来, 重头戏,
//            // (relative_double_image - noise_double_image) ^ 2 的积分图
//            for(int i = 0;i < length; ++i) {
//                const double temp = relative_double_image[i] - noise_double_image[i];
//                const double res = temp * temp;
//                residual_ptr[i] = res; // 均值滤波
//            }
//            // 算 box_filter
//            const auto residual_mean = box_filter(residual_ptr, radius, radius, H, W);
//            // 现在开始图像中的每一个点, 开始累计
//            for(int i = 0;i < length; ++i) {
//                const double cur_weight = std::exp(-residual_mean[i] * sigma_2_inv);
//                weight_sum[i] += cur_weight;
//                sum_value[i] += cur_weight * relative_double_image[i];
//                if(cur_weight > weight_max[i]) weight_max[i] = cur_weight;
//            }
//        }
//    }
//    // 结束之后
//    std::cout << "length  " << length << std::endl;
//    for(int i = 0;i < length; ++i) sum_value[i] += weight_max[i] * noise_double_image[i];
//    for(int i = 0;i < length; ++i) weight_sum[i] += weight_max[i];
//    for(int i = 0;i < length; ++i) denoised.data[i] = cv::saturate_cast<uchar>(255 * (sum_value[i] / weight_sum[i]));
//    return denoised;
//}
