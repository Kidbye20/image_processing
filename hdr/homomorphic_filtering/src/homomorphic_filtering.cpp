// C++
#include <vector>
#include <iostream>
#include <cmath>
#include <assert.h>
#include <filesystem>
#include <unordered_set>
#include <unordered_map>
// OpenCV
#include <opencv2/opencv.hpp>



namespace {

    void run(const std::function<void()>& work=[]{}, const std::string message="") {
        auto start = std::chrono::steady_clock::now();
        work();
        auto finish = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
        std::cout << message << " " << duration.count() << " ms" <<  std::endl;
    }

    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REFLECT);
        return padded_image;
    }

    inline float _min(const float* data, const int length) {
        float min_value = data[0];
        for(int i = 1;i < length; ++i)
            if(data[i] < min_value) min_value = data[i];
        return min_value;
    }

    inline float _max(const float* data, const int length) {
        float max_value = data[0];
        for(int i = 1;i < length; ++i)
            if(data[i] > max_value) max_value = data[i];
        return max_value;
    }

    inline float square(const float x) {
        return x * x;
    }

    inline float clip(float x, const float low, const float high) {
        if(x < low) x = low;
        else if(x > high) x = high;
        return x;
    }

    inline float fast_exp(const float y) {
        float d;
        *(reinterpret_cast<int*>(&d) + 0) = 0;
        *(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * y + 1072632447);
        return d;
    }

}




int main() {
	// 读取图像
	cv::Mat hdr_image = cv::imread("./images/input/vinesunset_2.hdr", cv::IMREAD_ANYDEPTH);


    return 0;
}











