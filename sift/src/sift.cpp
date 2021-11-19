//C++
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
// opencv
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>


namespace crane {
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

	cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
		cv::Mat padded_image;
		cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
		return padded_image;
	}

	cv::Mat cv_concat(const cv::Mat& lhs, const cv::Mat& rhs) {
        cv::Mat result;
        cv::hconcat(std::vector<cv::Mat>({lhs, rhs}), result);
        return result;
    }
}


cv::Mat gaussi_filter_channel(const cv::Mat& noise_channel, const int kernel_size, const double variance) {
    // 准备一些信息
    const int H = noise_channel.rows;
    const int W = noise_channel.cols;
    const int C = noise_channel.channels();
    if(C not_eq 1) {
        std::cout << "该函数只接受单通道图像的高斯滤波!" << std::endl;
        return noise_channel;
    }
    if(kernel_size % 2 == 0 or kernel_size <= 0) {
        std::cout << "滤波核的大小非法, 应当为正奇数!" << std::endl;
        return noise_channel;
    }
    // 算出半径大小
    const int radius = (kernel_size - 1) >> 1;
    // 给边缘做填充  // 256 x (288 + 5 x 2) x 1
    const auto padded_channel = crane::make_pad(noise_channel, radius, radius);
    // 先计算一个滤波模板, 一维的加速计算
    const int window_size = (radius << 1) + 1;
    std::vector<double> space_table(window_size * window_size, 0.0);
    std::vector<int> space_offset(window_size * window_size, 0);
    // 用指针访问, 内存释放交给 C++ 了
    double* const table_ptr = space_table.data();
    int* const offset_ptr = space_offset.data();
    // 相对位置的偏移
    int offset = 0;
    // e^{} 里面那个常数, 1. / (2 * δ * δ)
    const double variance_2 = -0.5 / (variance * variance);
    // 模板里的权重之和也是常数, 因为只跟核有关, 与图像无关
    double kernel_weight_sum = 0.0;
    // 计算 space_offset, 第 i 行, 然后加上第 j 个
    for(int i = -radius; i <= radius; ++i) {
        for(int j = -radius; j <= radius; ++j) {
            space_table[offset] = std::exp(variance_2 * (i * i + j * j));
            kernel_weight_sum += space_table[offset];
            space_offset[offset] = i * padded_channel.step + j;
            ++offset;
        }
    }
    // 乘法比除法快点, 这里可以快十五分之一
    double kernel_weight_sum_reverse = 1. / kernel_weight_sum;
    // 准备一个矩阵, 储存去噪的结果
    auto denoised_channel = noise_channel.clone();
    // 开始去噪
    for(int i = 0;i < H; ++i) {
        // 这里的这个 radius 行的 step 是一定要算的, 从 i = 0 开始算起, 随便打个草稿即可
        const uchar* const row_ptr = padded_channel.data + (radius + i) * padded_channel.step + radius;
        uchar* const row_ptr_denoise = denoised_channel.data + i * denoised_channel.step;
        for(int j = 0;j < W; ++j) {
            double value_sum = 0.0;
            // 遍历这个窗口
            for(int k = 0;k < offset; ++k) {
                const int pixel = row_ptr[j + offset_ptr[k]];
                const double weight = table_ptr[k];
                value_sum += weight * pixel;
            }
            // 这里的这个乘法特别占时间
            row_ptr_denoise[j] = cv::saturate_cast<uchar>(value_sum * kernel_weight_sum_reverse);
        }
    }
    return denoised_channel;
}

// 二维的高斯滤波可以用两个分离的一维高斯滤波来实现, 这个我可以试试怎么加速
// 时间复杂度 N x M x K^2



int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;
    // 根据图片路径读取图像
    const char* source_path = "./images/input/a0015-DSC_0081.png";
    const auto source_image = cv::imread(source_path);
    if(source_image.empty()) {
        std::cout << "读取图片  " << source_path << "  失败 !" << std::endl;
        return 0;
    }
    crane::cv_show(source_image);
    return 0;
}
