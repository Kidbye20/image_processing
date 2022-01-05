// C++
#include <vector>
#include <chrono>
#include <iostream>
#include <functional>
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

    void cv_info(const cv::Mat& one_image) {
        std::cout << "高  :  " << one_image.rows << "\n宽  :  " << one_image.cols << "\n通道 :  " << one_image.channels() << std::endl;
        std::cout << "步长 :  " << one_image.step << std::endl;
        std::cout << "是否连续" << std::boolalpha << one_image.isContinuous() << std::endl;
    }

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }
}



cv::Mat fast_gaussi_blur(
        const cv::Mat& source,
        const int radius=2,
        const float sigma=0.83,
        const float ratio=1.0,
        const bool mask=false) {
    // 根据半径和方差构造一个高斯模板
    const int filter_size = 2 * radius + 1;
    std::vector<float> filter(filter_size, 0);
    float cur_sum = 0;
    for(int i = -radius; i <= radius; ++i) {
        filter[i + radius] = 1.f / sigma * std::exp(-float(i * i) / (2 * sigma * sigma));
        cur_sum += filter[i + radius];
    }
    for(int i = 0;i < filter_size; ++i) filter[i] = filter[i] * ratio / cur_sum;
    // 先做 pad
    const auto source_pad = make_pad(source, radius, radius);
    // 获取图像信息
    const int H = source_pad.rows;
    const int W = source_pad.cols;
    const int C = source_pad.channels();
    // 先对 x 方向做高斯平滑
    cv::Mat temp = source_pad.clone();
    for(int ch = 0; ch < C; ++ch) {
        for(int i = 0; i < H; ++i) {
            const uchar* const src_ptr = source_pad.data + i * W * C;
            uchar* const temp_ptr = temp.data + i * W * C;
            for(int j = radius; j < W - radius; ++j) {
                // if(mask and src_ptr[j * C + ch] != 0) continue;
                float intensity_sum = 0;
                for(int k = -radius; k <= radius; ++k)
                    intensity_sum += filter[radius + k] * src_ptr[(j + k) * C + ch];
                temp_ptr[j * C + ch] = cv::saturate_cast<uchar>(intensity_sum);
            }
        }
    }
    // 再对 y 方向做高斯平滑
    cv::Mat result = source.clone();
    for(int ch = 0; ch < C; ++ch) {
        for(int i = radius; i < H - radius; ++i) {
            const uchar* const temp_ptr = temp.data + i * W * C;
            uchar* const res_ptr = result.data + (i - radius) * source.cols * C;
            for(int j = radius; j < W - radius; ++j) {
                // if(mask and temp_ptr[j * C + ch] != 0) continue;
                float intensity_sum = 0;
                for(int k = -radius; k <= radius; ++k)
                    intensity_sum += filter[radius + k] * temp_ptr[k * W * C + j * C + ch];
                res_ptr[(j - radius) * C + ch] = cv::saturate_cast<uchar>(intensity_sum);
            }
        }
    }
    return result;
}


cv::Mat pyramid_downsample(const cv::Mat& source) {
    // 收集图像信息
    const int H = source.rows / 2, W = source.cols / 2;
    // 准备一个结果
    cv::Mat downsampled(H, W, source.type());
    const int C = source.channels();
    // 开始每隔一个点采一个样
    for(int i = 0;i < H; ++i) {
        uchar* const res_ptr = downsampled.data + i * W * C;
        for(int j = 0;j < W; ++j)
            std::memcpy(res_ptr + j * C, source.data + 2 * (i * source.cols + j) * C, sizeof(uchar) * C);
    }
    return downsampled;
}


std::vector<cv::Mat> build_gaussi_pyramid(const cv::Mat& source, const int layers_num) {
    // 首先需要把图像规整到 2 ^ layers_num 的整数倍
    const int new_H = (1 << layers_num) * int(source.rows / (1 << layers_num));
    const int new_W = (1 << layers_num) * int(source.cols / (1 << layers_num));
    auto source_croped = source(cv::Rect(0, 0, new_W, new_H)).clone();
    // 准备返回结果
    std::vector<cv::Mat> gaussi_pyramid;
    gaussi_pyramid.reserve(layers_num);
    gaussi_pyramid.emplace_back(source_croped);
    // 开始构造接下来的几层
    for(int i = 1;i < layers_num; ++i) {
        // 先对图像做高斯模糊
        source_croped = fast_gaussi_blur(source_croped, 2, 1.0, 1.0);
        // 做下采样
        source_croped = pyramid_downsample(source_croped);
        // 放到高斯金字塔中
        gaussi_pyramid.emplace_back(source_croped);
    }
    return gaussi_pyramid;
}


cv::Mat pyramid_upsample(const cv::Mat& source) {
    const int H = source.rows, W = source.cols;
    const int C = source.channels();
    // 准备一个结果
    cv::Mat upsampled = cv::Mat::zeros(2 * H, 2 * W, source.type());
    // 把值填充到上采样结果中
    for(int i = 0; i < H; ++i) {
        const uchar* const src_ptr = source.data + i * W * C;
        uchar* const res_ptr = upsampled.data + 2 * i * (2 * W) * C;
        for(int j = 0;j < W; ++j)
            std::memcpy(res_ptr + 2 * j * C, src_ptr + j * C, sizeof(uchar) * C);
    }
    return upsampled;
}



std::vector<cv::Mat> build_laplace_pyramid(const std::vector<cv::Mat>& gaussi_pyramid) {
    // 查看几层
    const int layers_num = gaussi_pyramid.size();
    // 准备一个结果
    std::vector<cv::Mat> laplace_pyramid;
    laplace_pyramid.reserve(layers_num - 1);
    // 从低分辨率开始构建拉普拉斯金字塔
    for(int i = layers_num - 1; i >= 1; --i) {
        // 首先低分辨率先上采样到两倍大小
        cv::Mat upsampled = pyramid_upsample(gaussi_pyramid[i]);
        // 使用 4 倍的高斯滤波, 这个得格外写
        upsampled = fast_gaussi_blur(upsampled, 2, 1.0, 4.0, true);
        cv_show(upsampled);
    }
    return laplace_pyramid;
}


void laplace_decomposition_demo() {
    // 读取图像
    const std::string image_path("./images/input/a2376-IMG_2891.png");
    const std::string save_dir("./images/output/");
    cv::Mat origin_image = cv::imread(image_path);
    assert(!origin_image.empty() and "图片读取失败");
    // 根据图像构建高斯金字塔
    const auto gaussi_pyramid = build_gaussi_pyramid(origin_image, 5);
    // 构建拉普拉斯金字塔
    const auto laplace_pyramid = build_laplace_pyramid(gaussi_pyramid);
}



int main() {

    laplace_decomposition_demo();

    return 0;
}
