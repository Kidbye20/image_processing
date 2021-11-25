//C++
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
// self
#include "guided_filter.h"

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
        // 864 = 3 * 288
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
}


void denoise_gray_demo() {
    // 根据图片路径读取图像
    const char* noise_path = "../images/denoise/Kodak24/18.png";
    auto noise_image = cv::imread(noise_path);
    if(noise_image.empty()) {
        std::cout << "读取图片  " << noise_path << "  失败 !" << std::endl;
        return;
    }
    cv::cvtColor(noise_image, noise_image, cv::COLOR_BGR2GRAY);
    auto& guided_image = noise_image;
    const auto guided_result = guided_filter_channel_padded(noise_image, guided_image, 2, 2, 0.01);

    const auto comparison_results = cv_concat({noise_image, guided_result});
    cv_show(comparison_results);
    // 保存结果
    const std::string save_path("./results/comparison_gray.png");
    cv::imwrite(save_path, comparison_results, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}


void denoise_rgb_demo_1() {
    // 根据图片路径读取图像
    const char* noise_path = "../images/denoise/Kodak24/3.png";
    const auto noise_image = cv::imread(noise_path);
    if(noise_image.empty()) {
        std::cout << "读取图片  " << noise_path << "  失败 !" << std::endl;
        return;
    }
    // 自己对自己做引导滤波就是去噪, 那么此时, 将图像根据通道分离
    std::vector<cv::Mat> noise_channels;
    cv::split(noise_image, noise_channels);
    // 用一个 vector 接收每个通道的去噪结果
    std::vector<cv::Mat> denoised_channels;
    for(const auto& channel : noise_channels)
        denoised_channels.emplace_back(guided_filter_channel_padded(channel, channel, 2, 2, 0.01));
    // 把去噪结果合并起来
    cv::Mat denoised_image;
    cv::merge(denoised_channels, denoised_image);
    // 保存结果
    const auto comparison_results = cv_concat({noise_image, denoised_image});
    cv_show(comparison_results);
    const std::string save_path("./images/output/comparison_rgb.png");
    cv::imwrite(save_path, comparison_results, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}


void denoise_rgb_demo_2() {
    // 根据图片路径读取图像
    const char* noise_path = "../images/denoise/Kodak24/8.png";
    const auto noise_image = cv::imread(noise_path);
    if(noise_image.empty()) {
        std::cout << "读取图片  " << noise_path << "  失败 !" << std::endl;
        return;
    }
    // 自己对自己做引导滤波就是去噪, 那么此时, 将图像根据通道分离
    std::vector<cv::Mat> noise_channels;
    cv::split(noise_image, noise_channels);
    // 存储不同结果
    std::vector<cv::Mat> all_denoised_results({noise_image});
    std::vector<double> all_epsilons({0.0001, 0.001, 0.01, 0.1, 1});
    for(const auto epsilon : all_epsilons) {
        // 用一个 vector 接收每个通道的去噪结果
        std::vector<cv::Mat> denoised_channels;
        for(const auto& channel : noise_channels)
            denoised_channels.emplace_back(guided_filter_channel_padded(channel, channel, 2, 2, epsilon));
        // 把去噪结果合并起来
        cv::Mat denoised_image;
        cv::merge(denoised_channels, denoised_image);
        all_denoised_results.emplace_back(denoised_image);
    }
    // 保存结果
    const auto comparison_results = cv_concat(all_denoised_results);
    cv_show(comparison_results);
    const std::string save_path("./images/output/comparison_rgb_epsilon.png");
    cv::imwrite(save_path, comparison_results, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}


void denoise_rgb_demo_3() {
    // 根据图片路径读取图像
    const char* noise_path = "../images/denoise/Kodak24/22.png";
    const auto noise_image = cv::imread(noise_path);
    if(noise_image.empty()) {
        std::cout << "读取图片  " << noise_path << "  失败 !" << std::endl;
        return;
    }
    // 自己对自己做引导滤波就是去噪, 那么此时, 将图像根据通道分离
    std::vector<cv::Mat> noise_channels;
    cv::split(noise_image, noise_channels);
    // 存储不同结果
    std::vector<cv::Mat> all_denoised_results({noise_image});
    std::vector<double> all_radius({1, 3, 5, 7, 9});
    for(const auto radius : all_radius) {
        // 用一个 vector 接收每个通道的去噪结果
        std::vector<cv::Mat> denoised_channels;
        for(const auto& channel : noise_channels)
            denoised_channels.emplace_back(guided_filter_channel_padded(channel, channel, radius, radius, 0.01));
        // 把去噪结果合并起来
        cv::Mat denoised_image;
        cv::merge(denoised_channels, denoised_image);
        all_denoised_results.emplace_back(denoised_image);
    }
    // 保存结果
    const auto comparison_results = cv_concat(all_denoised_results);
    cv_show(comparison_results);
    const std::string save_path("./images/output/comparison_rgb_radius.png");
    cv::imwrite(save_path, comparison_results, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}


int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;
    // denoise_gray_demo();
    // denoise_rgb_demo_1();
    denoise_rgb_demo_2();
    denoise_rgb_demo_3();
    return 0;
}
