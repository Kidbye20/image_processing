//C++
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
// opencv
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>


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

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }

    cv::Mat cv_resize(cv::Mat& one_image, const int height, const int width, const int _interpolation=cv::INTER_LINEAR) {
		cv::Mat result;
		cv::resize(one_image, result, cv::Size(width, height), 0, 0, _interpolation);
		return result;
	}

	cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
		cv::Mat padded_image;
		cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
		return padded_image;
	}

	cv::Mat cv_concat(std::vector<cv::Mat> sequence) {
        cv::Mat result;
        cv::hconcat(sequence, result);
        return result;
    }
}



struct keypoint_type {
    cv::Point pos;
    int size;
    float response;
    keypoint_type(const int x, const int y, const int _size, const float _response)
        : pos(x, y), size(_size), response(_response) {}
};

std::vector< std::vector<cv::Mat> > build_DOG_pyramid(const cv::Mat& source, const int S=3, const float sigma=1.6, const int min_size=4) {
    // 首先上采样得到第一张图
    cv::Mat first_image;
    cv::resize(source, first_image, {0, 0}, 2, 2, cv::INTER_LINEAR);
    const float first_sigma = std::sqrt(sigma * sigma - (2 * 0.5) * (2 * 0.5));
    cv::GaussianBlur(first_image, first_image, {0, 0}, first_sigma, first_sigma);
    // 计算有几组图像, 最小分辨率是 4x4
    const int min_length = std::min(first_image.rows, first_image.cols);
    const int octaves_num = std::floor(std::log(min_length) / std::log(2) - min_size);
    std::cout << "octaves_num  " << octaves_num << std::endl;
    // 计算任意一组图像的方差序列
    const int images_num = S + 3;
    const float k = std::pow(2, 1.f / S);
    std::vector<float> sigmas_list(images_num, 0);
    sigmas_list[0] = sigma;
    for(int i = 1; i < images_num; ++i) {
        const float temp = std::pow(k, i - 1) * sigma;
        sigmas_list[i] = std::sqrt((k * k - 1) * temp * temp);
    }
    // 共 octaves_num 组, 每组有 S + 3 张图像, 对应尺度逐一做高斯模糊
    std::vector< std::vector<cv::Mat> > gaussi_scaled_pyramid;
    cv::Mat cur_scale;
    for(int i = 0;i < octaves_num; ++i) {
        std::vector<cv::Mat> this_octave;
        this_octave.reserve(images_num);
        if(i == 0) cur_scale = first_image.clone(); // 如果是第一组图像
        else { // 否则, 直接取上一组的倒数第三张图像作为起始图像, 下采样为原来的一半, 相当于尺度 * 2
            const cv::Mat& refer = gaussi_scaled_pyramid[i - 1][images_num - 1 - 3];
            cv::resize(refer, cur_scale, {refer.cols / 2, refer.rows / 2}, 0, 0, cv::INTER_LINEAR);
        }
        this_octave.emplace_back(cur_scale.clone()); // 起始图像都已经做了高斯模糊了, first_image 和上一组的倒数第三张
        for(int j = 1; j < images_num; ++j) {
            cv::GaussianBlur(cur_scale, cur_scale, {0, 0}, sigmas_list[j], sigmas_list[j]);
            this_octave.emplace_back(cur_scale.clone());
        }
        gaussi_scaled_pyramid.emplace_back(this_octave);
    }

    // 得到高斯差分金字塔
    std::vector< std::vector<cv::Mat> > DOG_pyramid;
    for(int i = 0;i < octaves_num; ++i) {
        std::vector<cv::Mat> this_octave;
        this_octave.reserve(images_num - 1);
        for(int j = 1;j < images_num; ++j)
            this_octave.emplace_back(gaussi_scaled_pyramid[i][j] - gaussi_scaled_pyramid[i][j - 1]);
        DOG_pyramid.emplace_back(this_octave);
//        std::cout << this_octave[0].row(0) << std::endl;
//        cv::Mat temp;
//        this_octave[0].convertTo(temp, CV_8UC1);
//        cv_show(temp);
    }
    return DOG_pyramid;
}

std::vector<keypoint_type> sift_detect_keypoints(
        const cv::Mat& _source,
        const int S=3,
        const float sigma=1.6,
        const int min_size=4,
        const float contrast_threshold=0.04) {
    // 转化成灰度图, 类型 float
    cv::Mat source;
    cv::cvtColor(_source, source, cv::COLOR_BGR2GRAY);
    assert(source.channels() == 1);
    source.convertTo(source, CV_32FC1);
    // 首先构建差分金字塔
    const auto DOG_pyramid = build_DOG_pyramid(source, S, sigma, min_size);
    // 寻找尺度空间极值 (x, y, sigma)
    const float threshold = std::floor(0.5 * contrast_threshold / S * 255);
    std::cout << "threshold  " << threshold << std::endl;
    std::vector<keypoint_type> keypoints;
    const int octaves_num = DOG_pyramid.size();
    const int images_num = DOG_pyramid[0].size();
    for(int o = 0;o < octaves_num; ++o) {
        for(int s = 1;s < images_num - 1; ++s) {
            // 获取上中下三组图像
            const auto& down_image = DOG_pyramid[o][s - 1];
            const auto& mid_image = DOG_pyramid[o][s];
            const auto& up_image = DOG_pyramid[o][s + 1];
            const int H = mid_image.rows, W = mid_image.cols;
            const int H_1 = H - 1, W_1 = W - 1;
            // 判断每个点是不是局部极值, mid[i][j] 和
            for(int i = 1;i < H_1; ++i) {
                const float* const down = down_image.ptr<float>() + i * W;
                const float* const mid = mid_image.ptr<float>() + i * W;
                const float* const up = up_image.ptr<float>() + i * W;
                for(int j = 1;j < W_1; ++j) {
                    const float center = std::abs(mid[j]);
                    if(center < threshold)
                        continue;
                    if(center > std::abs(mid[j - 1]) and center > std::abs(mid[j + 1]) and
                       center > std::abs(mid[j - 1 - W]) and center > std::abs(mid[j - W]) and center > std::abs(mid[j + 1 - W]) and
                       center > std::abs(mid[j - 1 + W]) and center > std::abs(mid[j + W]) and center > std::abs(mid[j + 1 + W]) and
                       center > std::abs(down[j - 1]) and center > std::abs(down[j]) and center > std::abs(down[j + 1]) and
                       center > std::abs(down[j - 1 - W]) and center > std::abs(down[j - W]) and center > std::abs(down[j + 1 - W]) and
                       center > std::abs(down[j - 1 + W]) and center > std::abs(down[j + W]) and center > std::abs(down[j + 1 + W]) and
                       center > std::abs(up[j - 1]) and center > std::abs(up[j]) and center > std::abs(up[j + 1]) and
                       center > std::abs(up[j - 1 - W]) and center > std::abs(up[j - W]) and center > std::abs(up[j + 1 - W]) and
                       center > std::abs(up[j - 1 + W]) and center > std::abs(up[j + W]) and center > std::abs(up[j + 1 + W])) {
                        const float temp = std::pow(2, o - 1);
                        const int size = sigma * std::pow(2, s / S)  * temp * 2; // 1.414
                        keypoints.emplace_back(j * temp, i * temp, size, center);
                    }
                }
            }
        }
    }
    std::cout << "收集到 " << keypoints.size() << " 个点\n";
    return keypoints;
}


int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;
    // 根据图片路径读取图像
    const char* source_path = "./images/input/a4644-Duggan_090214_5136.png";
    const auto source_image = cv::imread(source_path);
    assert(not source_image.empty() and "读取图像失败 !");
    // sift 检测关键点
    const auto keypoints = sift_detect_keypoints(source_image);
    // 展示与保存
    auto display = source_image.clone();
    for(const auto& point : keypoints)
        cv::circle(display, point.pos, point.size, CV_RGB(255, 0, 0), 1);
    cv_show(display);
    cv_write(display, "./images/output/keypoints_1.png");
    return 0;
}
