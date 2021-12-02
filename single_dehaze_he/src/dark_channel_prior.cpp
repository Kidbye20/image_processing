// C++
#include <assert.h>
#include <vector>
#include <iostream>
// self
#include "dark_channel_prior.h"

namespace {
    cv::Mat make_pad(const cv::Mat &one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }

    void cv_show(const cv::Mat &one_image, const char *info = "") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}


// 这里好像用 uchar 比较就行了 ?
cv::Mat get_dark_channel(const cv::Mat& I, const int H, const int W, const int radius) {
    // 准备一些变量
    const uchar* const I_ptr = I.data;
    // 暗通道
    cv::Mat dark_channel = cv::Mat::zeros(H, W, CV_8UC1);
    // 首先, 对于图像中每个像素, 我要比较一下 3 个通道, 记录每个像素在 3 个通道中的最小值
    const int length = H * W;
    for(int i = 0;i < length; ++i) {
        const int _beg = 3 * i;
        uchar min_pixel = I_ptr[_beg];
        if(min_pixel > I_ptr[_beg + 1]) min_pixel = I_ptr[_beg + 1];
        if(min_pixel > I_ptr[_beg + 2]) min_pixel = I_ptr[_beg + 2];
        dark_channel.data[i] = min_pixel;
    }
    auto min_dark_channel = dark_channel.clone();
    // 然后, 对这个暗通道执行区域最小值滤波
    dark_channel = make_pad(dark_channel, radius, radius);
    const int W2 = dark_channel.cols;
    const int kernel_size = (radius << 1) + 1;
    // 开始遍历每一个点
    for(int i = 0;i < H; ++i) {
        uchar* const min_row_ptr = min_dark_channel.data + i * W;
        for(int j = 0;j < W; ++j) {
            uchar min_pixel = 255;
            // 每个点, 找出 kernel, 然后逐一比较
            for(int x = 0;x < kernel_size; ++x) {
                // 获取 kernel 这一行的指针
                uchar* const row_ptr = dark_channel.data + (i + x) * W2 + j;
                for(int y = 0;y < kernel_size; ++y) if(min_pixel > row_ptr[y]) min_pixel = row_ptr[y];
            }
            min_row_ptr[j] = min_pixel;
        }
    }
    return min_dark_channel;
}


std::vector<double> get_global_atmospheric_light(const cv::Mat& I, const cv::Mat& I_dark, const int H, const int W, const double top_percent) {
    // 从 dark_channel 中, 选像素值前 0.1%
    // 首先要把所有点看一遍 ? 排序一把, 但这种其实可以用桶排序加速, 但是我还要记住位置,
    std::vector<int> book(256, 0);
    std::vector<int> pos[256];
    const uchar* const dark_ptr = I_dark.data;
    const uchar* const I_ptr = I.data;
    const int length = H * W;
    // 遍历 dark_channel, 记录每一个像素出现的位置
    for(int i = 0;i < length; ++i) {
        const int pixel = dark_ptr[i];
        ++book[pixel];
        pos[pixel].emplace_back(i);
    }
    // 统计完毕, 现在筛选出前 0.01 的 dark channel 的点的位置
    const int border = top_percent * length;
    std::vector<int> max_n(border, 0);
    int cnt = 0, ok = 0;
    for(int i = 255; i >= 0; --i) {
        const int frequency = book[i];
        for(int j = 0;j < frequency; ++j) {
            max_n[cnt++] = pos[i][j];
            if(cnt == border - 1) { ok = 1; break; }
        }
        if(ok) break;
    }
    // 现在 max_k 里保存了前 0.01 最大值的位置, 返回一个值还是三个值 ?
    std::vector<double> max_A(3, 0);
    for(int i = 0;i < border; ++i) {
        // 对应 color 图像中的点
        const int p = 3 * max_n[i];
        // 从图像中取出
        max_A[0] += I_ptr[p];
        max_A[1] += I_ptr[p + 1];
        max_A[2] += I_ptr[p + 2];
    }
    for(int i = 0;i < 3; ++i) max_A[i] /= border;
    return max_A;
}


cv::Mat dark_channel_prior_dehaze(const cv::Mat& I, const int radius, const double top_percent, const double t0, const double omega, const bool guided) {
    // 获取图像信息
    const int H = I.rows;
    const int W = I.cols;
    const int C = I.channels();
    assert(C == 3);
    const int length_1 = H * W * 3;
    const int length_2 = H * W;
    // 首先求 I 的暗通道图 I_dark
    const auto I_dark = get_dark_channel(I, H, W, radius);
    // 然后求全局大气光 A, 一个数, 需要汇总前 top_percent 的像素, 然后求出来
    const auto A = get_global_atmospheric_light(I, I_dark, H, W, top_percent);
    // 现在我求出了三个通道的 A, 准备求 1 - I_dark / A
    for(int ch = 0;ch < 3; ++ch) std::cout << A[ch] << std::endl;
    // 分别求三个通道的折射率 T
    std::vector< std::vector<double> > T(3, std::vector<double>(length_2));
    const uchar* I_dark_ptr = I_dark.data;
    for(int ch = 0;ch < 3; ++ch) {
        double* const cur_T = T[ch].data();
        for(int i = 0;i < length_2; ++i)
            cur_T[i] = 1 - omega * double(I_dark_ptr[i] / A[ch]);
        for(int i = 0;i < length_2; ++i)
            if(cur_T[i] < 0.1) cur_T[i] = 0.1;
    }
    // A, T, I 都已经知道了, 现在开始求
    auto dehazed = I.clone();
    uchar* const dehazed_ptr = dehazed.data;
    for(int i = 0;i < length_2; ++i) {
        const int _beg = 3 * i;
        dehazed_ptr[_beg] = cv::saturate_cast<uchar>(double(I.data[_beg] - A[0]) / T[0][i] + A[0]);
        dehazed_ptr[_beg + 1] = cv::saturate_cast<uchar>(double(I.data[_beg + 1] - A[1]) / T[1][i] + A[1]);
        dehazed_ptr[_beg + 2] = cv::saturate_cast<uchar>(double(I.data[_beg + 2] - A[2]) / T[2][i] + A[2]);
    }
    return dehazed;
}
