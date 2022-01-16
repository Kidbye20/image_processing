//C++
#include <list>
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
            class cluster {
            public:
                const float norm_inv = 1. / 255;
                int pixel_num;
                cv::Mat mu;
                float max_eigenvalue;
                cv::Mat e;
                const uchar* const _img_ptr;
                std::vector<int> _pixels;
                std::vector<float> _W;
                cluster(const uchar* const img_ptr, std::vector<int>& pixels, std::vector<float>& W)
                        : pixel_num(pixels.size()), _img_ptr(img_ptr), _pixels(pixels), _W(W){
                    // 首先求这些像素的颜色均值
                    this->mu = cv::Mat(1, 3, CV_32FC1);
                    float* mu_ptr = this->mu.ptr<float>();
                    float weight_sum = 0.0;
                    for(int i = 0;i < pixel_num; ++i) {
                        const int pos = 3 * pixels[i]; // 获取像素位置
                        mu_ptr[0] += norm_inv * img_ptr[pos] * W[i];
                        mu_ptr[1] += norm_inv * img_ptr[pos + 1] * W[i];
                        mu_ptr[2] += norm_inv * img_ptr[pos + 2] * W[i];
                        weight_sum += W[i];
                    }
                    for(int ch = 0; ch < 3; ++ch) mu_ptr[ch] /= weight_sum;
                    // 求协方差矩阵
                    cv::Mat diff(pixel_num, 3, CV_32FC1);
                    float* const diff_ptr = diff.ptr<float>();
                    for(int i = 0;i < pixel_num; ++i) {
                        const int pos = 3 * pixels[i];
                        const float sw = std::sqrt(W[i]);
                        diff_ptr[3 * i] = sw * (norm_inv * img_ptr[pos] - mu_ptr[0]);
                        diff_ptr[3 * i + 1] = sw * (norm_inv * img_ptr[pos + 1] - mu_ptr[1]);
                        diff_ptr[3 * i + 2] = sw * (norm_inv * img_ptr[pos + 2] - mu_ptr[2]);
                    }
                    const auto cov = diff.t() * diff / weight_sum + 1e-5 * cv::Mat::eye({3, 3}, CV_32FC1);
                    // 求特征值和特征向量
                    cv::Mat eigen_values, eigen_vector;
                    cv::eigen(cov, eigen_values, eigen_vector);
                    int max_index = 0;
                    max_eigenvalue = std::abs(eigen_values.at<double>(0));
                    for(int i = 1;i < eigen_values.rows; ++i) {
                        const float cur = std::abs(eigen_values.at<double>(i));
                        if(max_eigenvalue < cur) {
                            max_index = i;
                            max_eigenvalue = cur;
                        }
                    }
                    std::cout << eigen_vector << std::endl;
                    this->e = eigen_vector.colRange(max_index - 1, max_index).clone();
                    std::cout << "e: " << e << std::endl;
                }
                bool operator<(const cluster& rhs) const {
                    return this->max_eigenvalue < rhs.max_eigenvalue;
                }
                std::list<cluster> split() {
                    const float* const mu_ptr = this->mu.ptr<float>();
                    const float* const e_ptr = this->e.ptr<float>();
                    float threshold = 0;
                    for(int ch = 0;ch < 3; ++ch) threshold += mu_ptr[ch] * e_ptr[ch];
                    // 遍历所有像素,
                    std::vector<int> lhs, rhs;
                    std::vector<float> lhs_w, rhs_w;
                    for(int i = 0;i < this->pixel_num; ++i) {
                        const int pos = 3 * this->_pixels[i];
                        float dis = 0;
                        for(int ch = 0;ch < 3; ++ch)
                            dis += norm_inv * this->_img_ptr[pos] * e_ptr[ch];
                        if(dis < threshold) {
                            lhs.emplace_back(this->_pixels[i]);
                            lhs_w.emplace_back(this->_W[i]);
                        } else {
                            rhs.emplace_back(this->_pixels[i]);
                            rhs_w.emplace_back(this->_W[i]);
                        }
                    }
                    std::cout << lhs.size() << ",  " << rhs.size() << std::endl;
                    std::list<cluster> result;
                    result.emplace_back(cluster(this->_img_ptr, lhs, lhs_w));
                    result.emplace_back(cluster(this->_img_ptr, rhs, rhs_w));
                    return result;
                }
            };
            // 根据收集到的点 f_pixels 和 权重 f_weights
            cluster fore_cluster(foreground.data, f_pixels, f_weights);

            std::vector<cluster> nodes({fore_cluster});

            while(true) {
                // 获取当前所有簇中特征值最大的
                auto max_index = std::max_element(nodes.begin(), nodes.end()) - nodes.begin();
                std::cout << "max_index  " << max_index << std::endl;
//                if(nodes[max_index].max_eigenvalue > 0.05) {
                    // 对当前 nodes[max_index] 做分裂
                    auto& origin_cluster = nodes[max_index];
//
                    auto split_result = origin_cluster.split();
                    for(auto& item : split_result)
                        nodes.emplace_back(item);
//                    nodes.erase(nodes.begin() + max_index);
//                }
//                else break;
                break;
            }

            std::vector< std::pair<std::vector<float>, cv::Mat> > fore_cluster_results;
            for(const auto& one_cluster : nodes) {
                fore_cluster_results.emplace_back(one_cluster.mu, one_cluster.e);
            }

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
