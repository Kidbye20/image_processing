//C++
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
// Eigen3
#include <Eigen/Core>
#include <Eigen/Dense>
//OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
// self

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


    cv::Mat cv_concat(const std::vector<cv::Mat> images, const bool v=false) {
        cv::Mat result;
        if(not v) cv::hconcat(images, result);
        else cv::vconcat(images, result);
        return result;
    }

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_CONSTANT, 0);
        return padded_image;
    }

    template<typename T>
    cv::Mat toint8(const std::vector<T>& source, const int H, const int W, const int C, const int _type) {
        cv::Mat result(H, W, _type);
        const int length = H * W * C;
        for(int i = 0;i < length; ++i) result.data[i] = cv::saturate_cast<uchar>(std::abs(source[i]));
        return result;
    }
}



namespace {
    void find_unknown_edge(cv::Mat& cur_unknown, const int H, const int W) {
        cv::Mat new_cur_unknown = make_pad(cur_unknown, 1, 1);
        const int W2 = new_cur_unknown.cols;
        for(int i = 0; i < H; ++i) {
            uchar* const row_ptr = new_cur_unknown.data + (1 + i) * W2 + 1;
            uchar* const res_ptr = cur_unknown.data + i * W;
            for(int j = 0; j < W; ++j) {
                if(row_ptr[j] == 255) {
                    if(row_ptr[j - 1] == 0 or row_ptr[j + 1] == 0 or
                       row_ptr[j - 1 - W2] == 0 or row_ptr[j - W2] == 0 or row_ptr[j + 1 - W2] == 0 or
                       row_ptr[j - 1 + W2] == 0 or row_ptr[j + W2] == 0 or row_ptr[j + 1 + W2] == 0)
                        res_ptr[j] = 0;
                }
            }
        }
    }
}


void bayers_matting(
        const cv::Mat& observation,
        const cv::Mat& trimap,
        const int radius=12) {
    // 处理异常
    assert(observation.channels() == 3 and trimap.channels() == 1 and "观测图像必须是 3 通道, Trimap 图必须是单通道!");
    assert(observation.rows == trimap.rows and observation.cols == trimap.cols and "观测图像和 trimap 尺寸不一致 !");
    // 获取指针
    const uchar* const trimap_ptr = trimap.data;
    // 获取图像信息
    const int H = observation.rows;
    const int W = observation.cols;
    const int length = H * W;
    // 根据 trimap 提取前景和背景, 未知区域
    cv::Mat fore_mask(H, W, CV_8UC1);
    cv::Mat back_mask(H, W, CV_8UC1);
    cv::Mat unknown_mask(H, W, CV_8UC1);
    uchar* const fm_ptr = fore_mask.data, * const bm_ptr = back_mask.data, * const um_ptr = unknown_mask.data;
    for(int i = 0;i < length; ++i) {
        if(trimap_ptr[i] == 0) bm_ptr[i] = 255;
        else if(trimap_ptr[i] == 255) fm_ptr[i] = 255;
        else um_ptr[i] = 255;
    }
    cv::Mat foreground(H, W, observation.type());
    cv::Mat background(H, W, observation.type());
    observation.copyTo(foreground, fore_mask);
    observation.copyTo(background, back_mask);
    // 根据 trimap 得到初始的 alpha
    cv::Mat alpha = fore_mask.clone() / 255;
    alpha.convertTo(alpha, CV_32FC1);
    float* const alpha_ptr = alpha.ptr<float>();
    for(int i = 0;i < length; ++i) if(um_ptr[i] == 255) alpha_ptr[i] = -1; // 标识这个是未知的
    // 获取要求解的点的数目
    int targets_num = 0;
    for(int i = 0;i < length; ++i) if(um_ptr[i] == 255) ++targets_num;
    std::cout << "待求解的点有  " << targets_num << " 个!\n";
    // 求解需要的中间变量
    int obtained = 0;
    cv::Mat cur_unknown = unknown_mask.clone();
    // 开始求解
    while(obtained < targets_num) {
        // 首先, 获取未知区域这一圈的边缘
        find_unknown_edge(cur_unknown, H, W);
        // cv_show(cur_unknown);
        // 腐蚀之后, 找到当前不是要求的
        std::vector<int> unknown_edge;
        for(int i = 0;i < length; ++i)
            if(cur_unknown.data[i] == 0 and um_ptr[i] == 255)
                unknown_edge.emplace_back(i); // it / (W - 2) - 1 << ", " << it % W - 1
        // 求解这一圈
        const int unknown_edge_size = unknown_edge.size();
        std::cout << "这一圈要求解的数目有 " << unknown_edge_size << std::endl;
        for(int u = 0; u < unknown_edge_size; ++u) {
            // 获取当前要求解的点的坐标
            const int pos = unknown_edge[u];
            const int y = pos / W;
            const int y_min = std::max(0, y - radius);
            const int y_max = std::min(y + radius, H - 1);
            const int x = pos % W;
            const int x_min = std::max(0, x - radius);
            const int x_max = std::min(x + radius, W - 1);
            // 在这个区间范围内, 在 alpha 中找到不是 unknown 的点
            const int y_dis = y_max - y_min + 1;
            const int x_dis = x_max - x_min + 1;
            std::vector<float> window_alpha(y_dis * x_dis);
            for(int i = y_min; i <= y_max; ++i) {
                std::memcpy(window_alpha.data(), alpha_ptr + i * W + x_min, x_dis * sizeof(float));
            }
            // 找到周围 625 个点的 alpha
            // 找到周围 625 个点的 前景像素, 同时根据 alpha 求解 前景像素的权重
            // 找到周围 625 个点的 背景像素, 同时根据 alpha 求解 背景像素的权重
            // 如果范围内前景像素太少, 或者背景像素太少, 不求解
            // 对前景的有效像素聚类, 得到几个簇的颜色均值和方差
            // 对背景的有效像素聚类, 得到几个簇的颜色均值和方差
            // 获取这个局部区域内 alpha 的均值作为 alpha 的初始值
            // 迭代求解 F, B, 和 alpha
            // 得到的 F, B 分别赋值给 foreground 和 background
            // 得到的 alpha 赋值到对应位置
            // 标记这个点求解完毕 !
            um_ptr[pos] = 0;
            // 已求解数目 + 1
            ++obtained;
        }
    }
}



int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    // 读取图像
    const cv::Mat observation = cv::imread("./images/input/input_4.bmp");
    const cv::Mat trimap = cv::imread("./images/input/mask_4.bmp", cv::IMREAD_GRAYSCALE);
    assert(not observation.empty() and not trimap.empty() and "读取的图像为空 !");
    bayers_matting(observation, trimap);
    return 0;
}


/*
 * //C++
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
// Eigen3
#include <Eigen/Core>
#include <Eigen/Dense>
//OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
// self

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


    cv::Mat cv_concat(const std::vector<cv::Mat> images, const bool v=false) {
        cv::Mat result;
        if(not v) cv::hconcat(images, result);
        else cv::vconcat(images, result);
        return result;
    }

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_CONSTANT, 0);
        return padded_image;
    }

    template<typename T>
    cv::Mat toint8(const std::vector<T>& source, const int H, const int W, const int C, const int _type) {
        cv::Mat result(H, W, _type);
        const int length = H * W * C;
        for(int i = 0;i < length; ++i) result.data[i] = cv::saturate_cast<uchar>(std::abs(source[i]));
        return result;
    }
}



namespace {
    template<typename T>
    inline T square(const T x) {
        return x * x;
    }


    void find_unknown_edge(cv::Mat& cur_unknown, const int H, const int W) {
        cv::Mat new_cur_unknown = make_pad(cur_unknown, 1, 1);
        const int W2 = new_cur_unknown.cols;
        for(int i = 0; i < H; ++i) {
            uchar* const row_ptr = new_cur_unknown.data + (1 + i) * W2 + 1;
            uchar* const res_ptr = cur_unknown.data + i * W;
            for(int j = 0; j < W; ++j) {
                if(row_ptr[j] == 255) {
                    if(row_ptr[j - 1] == 0 or row_ptr[j + 1] == 0 or
                       row_ptr[j - 1 - W2] == 0 or row_ptr[j - W2] == 0 or row_ptr[j + 1 - W2] == 0 or
                       row_ptr[j - 1 + W2] == 0 or row_ptr[j + W2] == 0 or row_ptr[j + 1 + W2] == 0)
                        res_ptr[j] = 0;
                }
            }
        }
    }
}


void bayers_matting(
        const cv::Mat& _observation,
        const cv::Mat& _trimap,
        const int radius=12,
        const float sigma=8.0) {
    // 处理异常
    assert(_observation.channels() == 3 and _trimap.channels() == 1 and "观测图像必须是 3 通道, Trimap 图必须是单通道!");
    assert(_observation.rows == _trimap.rows and _observation.cols == _trimap.cols and "观测图像和 trimap 尺寸不一致 !");
    // 做 padding
    const auto observation = make_pad(_observation, radius, radius);
    const auto trimap = make_pad(_trimap, radius, radius);
    cv_write(observation, "./observation.png");
    cv_write(trimap, "./trimap.png");
    const uchar* const trimap_ptr = trimap.data;
    // 获取图像信息
    const int H = observation.rows;
    const int W = observation.cols;
    const int length = H * W;
    // 根据 trimap 提取前景和背景, 未知区域
    cv::Mat fore_mask(H, W, CV_8UC1);
    cv::Mat back_mask(H, W, CV_8UC1);
    cv::Mat unknown_mask(H, W, CV_8UC1);
    uchar* const fm_ptr = fore_mask.data, * const bm_ptr = back_mask.data, * const um_ptr = unknown_mask.data;
    for(int i = 0;i < length; ++i) {
        if(trimap_ptr[i] == 0) bm_ptr[i] = 255;
        else if(trimap_ptr[i] == 255) fm_ptr[i] = 255;
        else um_ptr[i] = 255;
    }
    cv::Mat foreground(H, W, observation.type());
    cv::Mat background(H, W, observation.type());
    observation.copyTo(foreground, fore_mask);
    observation.copyTo(background, back_mask);
    // 准备一个高斯核模板
    const int filter_size = square(radius * 2 + 1);
    std::vector<float> gaussi_filter(filter_size, 0);
    std::vector<int> gaussi_offset(filter_size, 0);
    int offset = 0;
    const float sigma_inv = 1. / (2 * sigma * sigma);
    for(int i = -radius; i <= radius; ++i) {
        for(int j = -radius; j <= radius; ++j) {
            gaussi_filter[offset] = std::exp(-1.f * (i * i + j * j) * sigma_inv);
            gaussi_offset[offset] = i * W + j;
            ++offset;
        }
    }
    float weight_sum = 0;
    for(int i = 0;i < filter_size; ++i) weight_sum += gaussi_filter[i];
    for(int i = 0;i < filter_size; ++i) gaussi_filter[i] /= weight_sum;
    // for(int i = 0;i < filter_size; ++i) std::cout << gaussi_filter[i] << std::endl;
    std::cout << "高斯核大小  " << filter_size << std::endl;
    // 根据 trimap 得到初始的 alpha
    cv::Mat alpha = fore_mask.clone() / 255;
    alpha.convertTo(alpha, CV_32FC1);
    float* const alpha_ptr = alpha.ptr<float>();
    for(int i = 0;i < length; ++i) if(um_ptr[i] == 255) alpha_ptr[i] = -1; // 标识这个是未知的
    // 获取要求解的点的数目
    int targets_num = 0;
    for(int i = 0;i < length; ++i) if(um_ptr[i] == 255) ++targets_num;
    std::cout << "待求解的点有  " << targets_num << " 个!\n";
    // 求解需要的中间变量
    int obtained = 1;
    cv::Mat cur_unknown = unknown_mask.clone();
    // 开始求解
    while(obtained < targets_num) {
        // 首先, 获取未知区域这一圈的边缘
        find_unknown_edge(cur_unknown, H, W);
        // cv_show(cur_unknown);
        // 腐蚀之后, 找到当前不是要求的
        std::vector<int> unknown_edge;
        for(int i = 0;i < length; ++i)
            if(cur_unknown.data[i] == 0 and um_ptr[i] == 255) // 被腐蚀的一圈和未求解的交集
                unknown_edge.emplace_back(i);
        // 求解这一圈
        const int unknown_edge_size = unknown_edge.size();
        // std::cout << "n  =  " << obtained << ", " << "um_ptr  =  " << unknown_edge.size() << std::endl;
        for(int u = 0; u < unknown_edge_size; ++u) {
            // 获取当前要求解的点的坐标
            const int pos = unknown_edge[u];
            const int y = pos / W;
            const int x = pos % W;
            std::cout << y << ", " << x << std::endl;
            if(u == 10) return;
            // 找到周围 625 个点的 前景像素
            std::vector<int> fore_pixels;
            std::vector<float> fore_weights;
            fore_pixels.reserve(filter_size);
            fore_weights.reserve(filter_size);
            for(int i = 0;i < filter_size; ++i) {
                const int index = pos + gaussi_offset[i];
                if(fm_ptr[index] == 255) { // 比较 255 要比较 8 位, 如果是 1 的话只要比较两位, 这里慢一点
                    fore_pixels.emplace_back(index); // 这个点是前景, 记录坐标
                    fore_weights.emplace_back(gaussi_filter[i] * alpha_ptr[index] * alpha_ptr[index]); // 计算这个点的权重
                }
            }
            if(fore_pixels.size() < 10) continue;
            // 找到周围 625 个点的 背景像素
            std::vector<int> back_pixels;
            std::vector<float> back_weights;
            back_pixels.reserve(filter_size);
            back_weights.reserve(filter_size);
            for(int i = 0;i < filter_size; ++i) {
                const int index = pos + gaussi_offset[i];
                if(bm_ptr[index] == 255) { // 比较 255 要比较 8 位, 如果是 1 的话只要比较两位, 这里慢一点
                    back_pixels.emplace_back(index); // 这个点是前景, 记录坐标
                    back_weights.emplace_back(gaussi_filter[i] * alpha_ptr[index] * alpha_ptr[index]); // 计算这个点的权重
                }
            }
            if(back_pixels.size() < 10) continue;
            // 找到周围 625 个点的 alpha
            // 同时根据 alpha 求解 前景像素的权重
            // 同时根据 alpha 求解 背景像素的权重
            // 如果范围内前景像素太少, 或者背景像素太少, 不求解
            // 对前景的有效像素聚类, 得到几个簇的颜色均值和方差
            // 对背景的有效像素聚类, 得到几个簇的颜色均值和方差
            // 获取这个局部区域内 alpha 的均值作为 alpha 的初始值
            // 迭代求解 F, B, 和 alpha
            // 得到的 F, B 分别赋值给 foreground 和 background
            // 得到的 alpha 赋值到对应位置

            // 标记这个点求解完毕 !
            um_ptr[pos] = 0;
            // 已求解数目 + 1
            ++obtained;
        }
    }
}



int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    // 读取图像
    const cv::Mat observation = cv::imread("./images/input/input_4.bmp");
    const cv::Mat trimap = cv::imread("./images/input/mask_4.bmp", cv::IMREAD_GRAYSCALE);
    assert(not observation.empty() and not trimap.empty() and "读取的图像为空 !");
    bayers_matting(observation, trimap);
    return 0;
}

 */

















/*
 *
 * //C++
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
// Eigen3
#include <Eigen/Core>
#include <Eigen/Dense>
//OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
// self

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


    cv::Mat cv_concat(const std::vector<cv::Mat> images, const bool v=false) {
        cv::Mat result;
        if(not v) cv::hconcat(images, result);
        else cv::vconcat(images, result);
        return result;
    }

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_CONSTANT, 0);
        return padded_image;
    }

    template<typename T>
    cv::Mat toint8(const std::vector<T>& source, const int H, const int W, const int C, const int _type) {
        cv::Mat result(H, W, _type);
        const int length = H * W * C;
        for(int i = 0;i < length; ++i) result.data[i] = cv::saturate_cast<uchar>(std::abs(source[i]));
        return result;
    }
}



namespace {
    template<typename T>
    inline T square(const T x) {
        return x * x;
    }


    void find_unknown_edge(cv::Mat& cur_unknown, const int H, const int W) {
        cv::Mat new_cur_unknown = make_pad(cur_unknown, 1, 1);
        const int W2 = new_cur_unknown.cols;
        for(int i = 0; i < H; ++i) {
            uchar* const row_ptr = new_cur_unknown.data + (1 + i) * W2 + 1;
            uchar* const res_ptr = cur_unknown.data + i * W;
            for(int j = 0; j < W; ++j) {
                if(row_ptr[j] == 255) {
                    if(row_ptr[j - 1] == 0 or row_ptr[j + 1] == 0 or
                       row_ptr[j - 1 - W2] == 0 or row_ptr[j - W2] == 0 or row_ptr[j + 1 - W2] == 0 or
                       row_ptr[j - 1 + W2] == 0 or row_ptr[j + W2] == 0 or row_ptr[j + 1 + W2] == 0)
                        res_ptr[j] = 0;
                }
            }
        }
    }
}


void bayers_matting(
        const cv::Mat& _observation,
        const cv::Mat& _trimap,
        const int radius=12,
        const float sigma=8.0) {
    // 处理异常
    assert(_observation.channels() == 3 and _trimap.channels() == 1 and "观测图像必须是 3 通道, Trimap 图必须是单通道!");
    assert(_observation.rows == _trimap.rows and _observation.cols == _trimap.cols and "观测图像和 trimap 尺寸不一致 !");
    // 做 padding
    const auto observation = make_pad(_observation, radius, radius);
    const auto trimap = make_pad(_trimap, radius, radius);
    cv_write(observation, "./observation.png");
    cv_write(trimap, "./trimap.png");
    const uchar* const trimap_ptr = trimap.data;
    // 获取图像信息
    const int H = observation.rows;
    const int W = observation.cols;
    const int length = H * W;
    // 根据 trimap 提取前景和背景, 未知区域
    cv::Mat fore_mask(H, W, CV_8UC1);
    cv::Mat back_mask(H, W, CV_8UC1);
    cv::Mat unknown_mask(H, W, CV_8UC1);
    uchar* const fm_ptr = fore_mask.data, * const bm_ptr = back_mask.data, * const um_ptr = unknown_mask.data;
    for(int i = 0;i < length; ++i) {
        if(trimap_ptr[i] == 0) bm_ptr[i] = 255;
        else if(trimap_ptr[i] == 255) fm_ptr[i] = 255;
        else um_ptr[i] = 255;
    }
    cv::Mat foreground(H, W, observation.type());
    cv::Mat background(H, W, observation.type());
    observation.copyTo(foreground, fore_mask);
    observation.copyTo(background, back_mask);
    // 准备一个高斯核模板
    const int filter_size = square(radius * 2 + 1);
    std::vector<float> gaussi_filter(filter_size, 0);
    std::vector<int> gaussi_offset(filter_size, 0);
    int offset = 0;
    const float sigma_inv = 1. / (2 * sigma * sigma);
    for(int i = -radius; i <= radius; ++i) {
        for(int j = -radius; j <= radius; ++j) {
            gaussi_filter[offset] = std::exp(-1.f * (i * i + j * j) * sigma_inv);
            gaussi_offset[offset] = i * W + j;
            ++offset;
        }
    }
    float weight_sum = 0;
    for(int i = 0;i < filter_size; ++i) weight_sum += gaussi_filter[i];
    for(int i = 0;i < filter_size; ++i) gaussi_filter[i] /= weight_sum;
    // for(int i = 0;i < filter_size; ++i) std::cout << gaussi_filter[i] << std::endl;
    std::cout << "高斯核大小  " << filter_size << std::endl;
    // 根据 trimap 得到初始的 alpha
    cv::Mat alpha = fore_mask.clone() / 255;
    alpha.convertTo(alpha, CV_32FC1);
    float* const alpha_ptr = alpha.ptr<float>();
    for(int i = 0;i < length; ++i) if(um_ptr[i] == 255) alpha_ptr[i] = -1; // 标识这个是未知的
    // 获取要求解的点的数目
    int targets_num = 0;
    for(int i = 0;i < length; ++i) if(um_ptr[i] == 255) ++targets_num;
    std::cout << "待求解的点有  " << targets_num << " 个!\n";
    // 求解需要的中间变量
    int obtained = 1;
    cv::Mat cur_unknown = unknown_mask.clone();
    // 开始求解
    while(obtained < targets_num) {
        // 首先, 获取未知区域这一圈的边缘
        find_unknown_edge(cur_unknown, H, W);
        // cv_show(cur_unknown);
        // 腐蚀之后, 找到当前不是要求的
        std::vector<int> unknown_edge;
        for(int i = 0;i < length; ++i)
            if(cur_unknown.data[i] == 0 and um_ptr[i] == 255) // 被腐蚀的一圈和未求解的交集
                unknown_edge.emplace_back(i);
        // 求解这一圈
        const int unknown_edge_size = unknown_edge.size();
        // std::cout << "n  =  " << obtained << ", " << "um_ptr  =  " << unknown_edge.size() << std::endl;
        for(int u = 0; u < unknown_edge_size; ++u) {
            // 获取当前要求解的点的坐标
            const int pos = unknown_edge[u];
            // 找到周围 625 个点的 alpha
            // 得到初始化的 alpha
            int valid_cnt = 0;
            float init_alpha = 0;
            std::vector<int> f_pixels, b_pixels;  // cv::Rect 更快一点
            std::vector<float> f_weights, b_weights;
            f_pixels.reserve(filter_size), b_pixels.reserve(filter_size);
            f_weights.reserve(filter_size), b_weights.reserve(filter_size);
            for(int i = 0;i < filter_size; ++i) {
                const int index = pos + gaussi_offset[i];
                if(alpha_ptr[index] != -1) { // 当前点不是未知点 NAN
                    // 前景
                    float cur_f_weight = square(alpha_ptr[index]) * gaussi_filter[i];
                    if(cur_f_weight > 0) {
                        f_pixels.emplace_back(index);
                        f_weights.emplace_back(cur_f_weight);
                    }
                    // 背景
                    float cur_b_weight = square(1 - alpha_ptr[index]) * gaussi_filter[i];
                    if(cur_b_weight > 0) {
                        b_pixels.emplace_back(index);
                        b_weights.emplace_back(cur_b_weight);
                    }
                    // 记录不是空的 alpla
                    init_alpha += alpha_ptr[index];
                    ++valid_cnt;
                }
            }
            init_alpha /= valid_cnt;
            if(f_weights.size() < 10 or b_weights.size() < 10)
                continue;
            std::cout << pos / W << ", " << pos % W << " ==> " << f_weights.size() << ", " << b_weights.size() << std::endl;

            // 对前景和后景进行聚类, 定义每个簇
            struct cluster {
                const float norm_inv = 1. / 255;
                cluster(const uchar* const img_ptr, const std::vector<int>& pixels, const std::vector<float>& W) {
                    // 首先求这些像素的颜色均值
                    float weight_sum = 0.0;
                    std::vector<float> mu(3, 0);
                    const int pixel_num = pixels.size();
                    for(int i = 0;i < pixel_num; ++i) {
                        const int pos = 3 * pixels[i]; // 获取像素位置
                        mu[0] += norm_inv * img_ptr[pos] * W[i];
                        mu[1] += norm_inv * img_ptr[pos + 1] * W[i];
                        mu[2] += norm_inv * img_ptr[pos + 2] * W[i];
                        weight_sum += W[i];
                    }
                    for(int ch = 0; ch < 3; ++ch) mu[ch] /= weight_sum;
                    // 求协方差矩阵
                    Eigen::MatrixXf diff(pixel_num, 3);
                    for(int i = 0;i < pixel_num; ++i) {
                        const int pos = 3 * pixels[i];
                        const float sw = std::sqrt(W[i]);
                        diff(i, 0) = sw * (norm_inv * img_ptr[pos] - mu[0]);
                        diff(i, 1) = sw * (norm_inv * img_ptr[pos + 1] - mu[1]);
                        diff(i, 2) = sw * (norm_inv * img_ptr[pos + 2] - mu[2]);
                    }
                    Eigen::Matrix<float, 3, 3> cov = diff.transpose() * diff / weight_sum;
                    cov += 1e-5 * Eigen::MatrixXf::Identity(3, 3);
                    // 解特征值
                    Eigen::EigenSolver< Eigen::Matrix<float, 3, 3> > eigen_solver(cov);
                    const auto& eigen_values = eigen_solver.eigenvalues().real();
                    std::cout << eigen_values << std::endl;
                    const auto& eigen_vectors = eigen_solver.eigenvectors().real();
                    std::cout << eigen_values(0, 0) << std::endl;
                    const int eigen_size = eigen_values.size();
                    std::vector<float> eigen_values_std({
                        abs(eigen_values(0)),
                        abs(eigen_values(1)),
                        abs(eigen_values(2))});
                    // 找最大的特征值和对应的
                    int max_index = 0;
                    float max_eigen_value = eigen_values_std[0];
                    for(int i = 1;i < eigen_size; ++i) {
                        if(max_eigen_value < eigen_values_std[i]) {
                            max_eigen_value = eigen_values_std[i];
                            max_index = i;
                        }
                    }
                    std::cout << "max_index  " << max_index << std::endl;
                    std::cout << eigen_vectors << std::endl;
                    Eigen::Vector3f e;
                    for(int ch = 0;ch < 3; ++ch)
                        e(ch) = eigen_vectors(max_index, ch);
                    std::cout << e << std::endl;
                }
            };
            // 根据收集到的点 f_pixels 和 权重 f_weights
            cluster fore_cluster(foreground.data, f_pixels, f_weights);

            // 定义
            return;


            // 迭代求解 F, B, 和 alpha
            // 得到的 F, B 分别赋值给 foreground 和 background
            // 得到的 alpha 赋值到对应位置

            // 标记这个点求解完毕 !
            um_ptr[pos] = 0;
            // 已求解数目 + 1
            ++obtained;
        }
        return;
    }
}



int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    // 读取图像
    const cv::Mat observation = cv::imread("./images/input/input_4.bmp");
    const cv::Mat trimap = cv::imread("./images/input/mask_4.bmp", cv::IMREAD_GRAYSCALE);
    assert(not observation.empty() and not trimap.empty() and "读取的图像为空 !");
    bayers_matting(observation, trimap);
    return 0;
}

 */