//C++
#include <map>
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
// self
#include "dark_channel_prior.h"

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

    cv::Mat cv_stack(const std::vector<cv::Mat> images) {
        cv::Mat result;
        cv::merge(images, result);
        return result;
    }
}



void dark_channel_prior_demo_1() {
    const std::string image_path("../images/dehaze/he_2019/tiananmen1.bmp");
    const auto haze_image = cv::imread(image_path);
    if(haze_image.empty()) {
        std::cout << "读取图像 " << image_path << " 失败 !\n";
        return;
    }
    // 送到暗通道先验去雾算法
    std::map<const std::string, cv::Mat> dehazed_result;
    run([&](){
        dehazed_result = dark_channel_prior_dehaze(haze_image, 3, 0.001, 0.1, 0.95, false);
    }, "普通 3 通道分别估计 T(x) ====>  ");
    // ---------- 【】展示结果与保存
    // const auto dark_channel = cv_stack({dehazed_result["dark_channel"], dehazed_result["dark_channel"], dehazed_result["dark_channel"]});
    const auto comparison_results = cv_concat({haze_image, dehazed_result["dehazed"]});
    cv_show(comparison_results);
    const std::string save_path("./images/output/comparison_1.png");
    cv::imwrite(save_path, comparison_results, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}



int main() {
    dark_channel_prior_demo_1();
    return 0;
}
