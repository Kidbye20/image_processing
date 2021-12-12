//C++
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// self
#include "faster_gaussi_filter.h"

namespace {
    void run(const std::function<void()>& work=[]{}, const std::string message="") {
        auto start = std::chrono::steady_clock::now();
        work();
        auto finish = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
        std::cout << message << " " << duration.count() << " ms" <<  std::endl;
    }

    void cv_info(const cv::Mat& one_image) {
        std::cout << "高  :  " << one_image.rows << "\n宽  :  " << one_image.cols << "\n通道 :  " << one_image.channels() << std::endl;
        std::cout << "步长 :  " << one_image.step << std::endl;
    }

    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    cv::Mat cv_resize(cv::Mat& one_image, const int height, const int width, const int _interpolation=cv::INTER_LINEAR) {
		cv::Mat result;
		cv::resize(one_image, result, cv::Size(width, height), 0, 0, _interpolation);
		return result;
	}

	cv::Mat cv_concat(const cv::Mat& lhs, const cv::Mat& rhs) {
        cv::Mat result;
        cv::hconcat(std::vector<cv::Mat>({lhs, rhs}), result);
        return result;
    }

    cv::Mat cv_concat(const std::vector<cv::Mat> images) {
        cv::Mat result;
        cv::hconcat(images, result);
        return result;
    }

    cv::Mat cv_repeat(const cv::Mat& source) {
        cv::Mat result;
        cv::merge(std::vector<cv::Mat>({source, source, source}), result);
        return result;
    }

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }

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





cv::Mat laplace_detail_enhance(const cv::Mat& source) {
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
            // const uchar u = row_ptr[j - W2], d = row_ptr[j + W2], l = row_ptr[j - 1], r = row_ptr[j + 1];
            // res_ptr[j] = cv::saturate_cast<uchar>(std::abs(u + d + l + r - 4 * row_ptr[j]));
            const uchar u = row_ptr[j - W2], d = row_ptr[j + W2], l = row_ptr[j - 1], r = row_ptr[j + 1];
            const uchar u_1 = row_ptr[j - W2], u_2 = row_ptr[j - 1], d_1 = row_ptr[j + W2], d_2 = row_ptr[j + 1];
            double value = u + d + l + r + u_1 + u_2 + d_1 + d_2 - 8 * row_ptr[j];
            if(value < 0) value = -value;
            res_ptr[j] = cv::saturate_cast<uchar>(value);
        }
    }
    return result;
}






int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    std::string noise_path("../images/detail/a0041-IMG_4972.png");
    auto noise_image = cv::imread(noise_path);
    if(noise_image.empty()) {
        std::cout << "读取图像 " << noise_path << " 失败 !" << std::endl;
        return 0;
    }
    //noise_image = cv_resize(noise_image, 512, 341);
    // 转成灰度图
    cv::Mat noise_gray;
    cv::cvtColor(noise_image, noise_gray, cv::COLOR_BGR2GRAY);
    // 先过一遍高斯滤波
    // noise_gray = faster_2_gaussi_filter_channel(noise_gray, 3, 0.1, 0.1);
    // 利用拉普拉斯检测边缘
    auto details = laplace_detail_enhance(noise_gray);
    cv_show(details);
    // 增强细节
    const auto comparison_results = cv_concat({noise_image, noise_image + cv_repeat(details)});
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/demo_2.png");
    return 0;
}
