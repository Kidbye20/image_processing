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
    }

    void cv_show(const cv::Mat& one_image, const char* info="") {
        // cv::namedWindow("", cv::WindowFlags::WINDOW_NORMAL);
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    cv::Mat cv_resize(cv::Mat& one_image, const int height, const int width, const int _interpolation=cv::INTER_LINEAR) {
		cv::Mat result;
		cv::resize(one_image, result, cv::Size(width, height), 0, 0, _interpolation);
		return result;
	}

	cv::Mat cv_concat(const cv::Mat& lhs, const cv::Mat& rhs, const bool v=false) {
        cv::Mat result;
        if(not v) cv::hconcat(std::vector<cv::Mat>({lhs, rhs}), result);
        else cv::vconcat(std::vector<cv::Mat>({lhs, rhs}), result);
        return result;
    }

    cv::Mat cv_concat(const std::vector<cv::Mat> images, const bool v=false) {
        cv::Mat result;
        if(not v) cv::hconcat(images, result);
        else cv::vconcat(images, result);
        return result;
    }

    cv::Mat cv_stack(const std::vector<cv::Mat> images) {
        cv::Mat result;
        cv::merge(images, result);
        return result;
    }

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }
}



void dark_channel_prior_demo_1() {
    const std::string image_path("../images/dehaze/he_2019/tiananmen1.bmp");
    const auto haze_image = cv::imread(image_path);
    if(haze_image.empty()) {
        std::cout << "读取图像 " << image_path << " 失败 !\n";
        return;
    }
    cv::Mat comparison_results;
    std::map<const std::string, cv::Mat> dehazed_result;

    // ---------- 【1】三通道一个透射图 T, 没有 guided filter 精修
    run([&](){
        dehazed_result = dark_channel_prior_dehaze(haze_image, 3, 0.001, 0.1, 0.95, false, false, true);
    }, "t(x) = 1, without guide  ====>  ");
    // 计算深度图 ?
    comparison_results = cv_concat({
        haze_image,
        cv_stack({dehazed_result["dark_channel"], dehazed_result["dark_channel"], dehazed_result["dark_channel"]}),
        cv_stack({dehazed_result["T"], dehazed_result["T"], dehazed_result["T"]}),
        dehazed_result["dehazed"]});
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/t_1_without_guide.png");

    // ---------- 【2】三通道一个透射图 T, 经过 guided filter 精修
    run([&](){
        dehazed_result = dark_channel_prior_dehaze(haze_image, 3, 0.001, 0.1, 0.95, true, false, true);
    }, "t(x) = 1, with guide  ====>  ");
    comparison_results = cv_concat({
        haze_image,
        cv_stack({dehazed_result["dark_channel"], dehazed_result["dark_channel"], dehazed_result["dark_channel"]}),
        cv_stack({dehazed_result["T"], dehazed_result["T"], dehazed_result["T"]}),
        cv_stack({dehazed_result["T_guided"], dehazed_result["T_guided"], dehazed_result["T_guided"]}),
        dehazed_result["dehazed"]});
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/t_1_with_guide.png");

    // ---------- 【3】三通道分开计算透射图 T, 不经过 guided filter 精修
    run([&](){
        dehazed_result = dark_channel_prior_dehaze(haze_image, 3, 0.001, 0.1, 0.95, false, true, true);
    }, "t(x) = 3, without guide  ====>  ");
    comparison_results = cv_concat(
        cv_concat({
                haze_image,
                cv_stack({dehazed_result["dark_channel"], dehazed_result["dark_channel"], dehazed_result["dark_channel"]}),
                dehazed_result["dehazed"]}),
        cv_concat({
                cv_stack({dehazed_result["T_0"], dehazed_result["T_0"], dehazed_result["T_0"]}),
                cv_stack({dehazed_result["T_1"], dehazed_result["T_1"], dehazed_result["T_1"]}),
                cv_stack({dehazed_result["T_2"], dehazed_result["T_2"], dehazed_result["T_2"]})
        }),
        true
    );
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/t_3_without_guide.png");

    // ---------- 【4】三通道分开计算透射图 T, 经过 guided filter 精修
    run([&](){
        dehazed_result = dark_channel_prior_dehaze(haze_image, 3, 0.001, 0.1, 0.95, true, true, true);
    }, "t(x) = 3, with guide  ====>  ");
    comparison_results = cv_concat(
        {
            cv_concat({
                haze_image,
                cv_stack({dehazed_result["dark_channel"], dehazed_result["dark_channel"], dehazed_result["dark_channel"]}),
                dehazed_result["dehazed"]}),
            cv_concat({
                    cv_stack({dehazed_result["T_0"], dehazed_result["T_0"], dehazed_result["T_0"]}),
                    cv_stack({dehazed_result["T_1"], dehazed_result["T_1"], dehazed_result["T_1"]}),
                    cv_stack({dehazed_result["T_2"], dehazed_result["T_2"], dehazed_result["T_2"]})}),
            cv_concat({
                    cv_stack({dehazed_result["T_0_guided"], dehazed_result["T_0_guided"], dehazed_result["T_0_guided"]}),
                    cv_stack({dehazed_result["T_1_guided"], dehazed_result["T_1_guided"], dehazed_result["T_1_guided"]}),
                    cv_stack({dehazed_result["T_2_guided"], dehazed_result["T_2_guided"], dehazed_result["T_2_guided"]})})
        },
        true
    );
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/t_3_with_guide.png");
}


void dark_channel_prior_demo_2() {
    const std::string image_path("../images/dehaze/he_2019/train.bmp");
    const auto haze_image = cv::imread(image_path);
    if(haze_image.empty()) {
        std::cout << "读取图像 " << image_path << " 失败 !\n";
        return;
    }
    cv::Mat comparison_results;
    std::map<const std::string, cv::Mat> dehazed_result;

    // ---------- 【1】
    run([&](){
        dehazed_result = dark_channel_prior_dehaze(haze_image, 3, 0.001, 0.1, 0.95, true, false, true);
    }, "t(x) = 1, with guide  ====>  ");

    // 制作热度图
    // 随便选一个 T, 我选精修之后的, 原始的 T 也可以(注意这里取键值, 必须取有键值的, 不然程序崩溃)
    const auto& T = dehazed_result["T_guided"];
    cv::Mat T_double;
    T.convertTo(T_double, CV_64FC1);
    cv::log(T_double, T_double);
    T_double = - 1. / 0.1 * T_double;
    cv::Mat hotmap_image = cv::Mat::zeros(T.rows, T.cols, CV_64FC1);
    cv::normalize(T_double, hotmap_image, 0, 255, cv::NORM_MINMAX);
    hotmap_image.convertTo(hotmap_image, CV_8UC1);
    cv::Mat temp;
    cv::applyColorMap(hotmap_image, temp, cv::COLORMAP_HOT);
    cv::cvtColor(hotmap_image, hotmap_image, cv::COLOR_BGR2RGB);
    cv::addWeighted(hotmap_image, 0.1, temp, 0.9, 0, hotmap_image);

    comparison_results = cv_concat(
            {
        cv_concat({
            haze_image,
            cv_stack({dehazed_result["dark_channel"], dehazed_result["dark_channel"], dehazed_result["dark_channel"]}),
            hotmap_image
        }),
        cv_concat({
            cv_stack({dehazed_result["T"], dehazed_result["T"], dehazed_result["T"]}),
            cv_stack({T, T, T}),
            dehazed_result["dehazed"]
        }),
    }, true);
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/t_1_with_guide_hotmap.png");

    // 对比
    std::map<const std::string, cv::Mat> dehazed_result_2;
    run([&](){
        dehazed_result_2 = dark_channel_prior_dehaze(haze_image, 3, 0.001, 0.1, 0.95, false, false, true);
    }, "t(x) = 1, without guide  ====>  ");
    // 中间留一个空白区域
    cv::Mat blank = cv::Mat(cv::Size(40, T.rows), CV_8UC3);
    const int TOTAL = 3 * 40 * T.rows;
    for(int i = 0;i < TOTAL; ++i) blank.data[i] = 255;
    comparison_results = cv_concat({dehazed_result_2["dehazed"], blank, dehazed_result["dehazed"]});
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/t_1_before_and_after_guide_fiilter.png");
}


int main() {
    // 最开始的探索
    // dark_channel_prior_demo_1();

    // 简易版
    dark_channel_prior_demo_2();
    return 0;
}
