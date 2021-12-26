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

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
		cv::Mat padded_image;
		cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
		return padded_image;
	}

    cv::Mat cv_concat(const std::vector<cv::Mat> images, const bool v=false) {
        cv::Mat result;
        if(not v) cv::hconcat(images, result);
        else cv::vconcat(images, result);
        return result;
    }

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }

    template<typename T>
    cv::Mat toint8(const std::vector<T>& source, const int H, const int W) {
        cv::Mat result(H, W, CV_8UC1);
        const int length = H * W;
        for(int i = 0;i < length; ++i) result.data[i] = std::abs(source[i]);
        return result;
    }
}



void box_filter(int* const source, double* const target, const int radius, const int H, const int W) {
    // 储存这一行的结果
    const int window_size = (radius << 1) + 1;
    std::vector<double> buffer(W);
    // 遍历前 window_size 行, 计算每一列的和
    for(int i = 0; i < window_size; ++i) {
        int* const row_ptr = source + i * W;
        for(int j = 0; j < W; ++j) buffer[j] += row_ptr[j];
    }
    // 计算剩下每一行的
    const int H2 = H - radius;
    const int W2 = W - 2 * radius;
    for(int i = radius;i < H2; ++i) {
        // 计算第一个位置的和
        double cur_sum = 0;
        for(int j = 0;j < window_size; ++j) cur_sum += buffer[j];
        // 记录这第一个位置的结果
        const int _beg = i * W + radius;
        target[_beg] = cur_sum;
        // 开始向右挪动
        for(int j = 1; j < W2; ++j) {
            cur_sum = cur_sum - buffer[j - 1] + buffer[j - 1 + window_size];
            target[_beg + j] = cur_sum;
        }
        // 这一行移动完毕, 换到下一行, 更新 buffer
        if(i != H2 - 1) {
            int* up_ptr = source + (i - radius) * W;
            int* down_ptr = source + (i + radius + 1) * W;
            for(int j = 0;j < W; ++j) buffer[j] = buffer[j] - up_ptr[j] + down_ptr[j];
        }
    }
}


using key_points_type = std::vector< std::tuple<double, int, int> >;
key_points_type harris_corner_detect(
        const cv::Mat& source,
        const int radius=2,
        const double alpha=0.04,
        const double threshold=1e5,
        const int point_num=-1) {
    // 获取图像信息
    const int H = source.rows;
    const int W = source.cols;
    const int length = H * W;
    // 先获取 x, y 两个方向上的梯度, 这里用的是 Sobel
    std::vector<int> gradients_x(length, 0);
    std::vector<int> gradients_y(length, 0);
    const int H_1 = H - 1, W_1 = W - 1;
    for(int i = 1; i < H_1; ++i) {
        const uchar* const row_ptr = source.data + i * W;
        int* const x_ptr = gradients_x.data() + i * W;
        int* const y_ptr = gradients_y.data() + i * W;
        for(int j = 1; j < W_1; ++j) {
            // 计算 Sobel 梯度
            x_ptr[j] = 2 * row_ptr[j + 1] + row_ptr[j + 1 + W] + row_ptr[j + 1 - W] - (2 * row_ptr[j - 1] + row_ptr[j - 1 + W] + row_ptr[j - 1 - W]);
            y_ptr[j] = 2 * row_ptr[j + W] + row_ptr[j + W + 1] + row_ptr[j + W - 1] - (2 * row_ptr[j - W] + row_ptr[j - W + 1] + row_ptr[j - W - 1]);
        }
    }
    // 计算 xx, yy, xy
    std::vector<int> gradients_xx(length, 0), gradients_yy(length, 0), gradients_xy(length, 0);
    for(int i = 0;i < length; ++i) gradients_xx[i] = gradients_x[i] * gradients_x[i];
    for(int i = 0;i < length; ++i) gradients_yy[i] = gradients_y[i] * gradients_y[i];
    for(int i = 0;i < length; ++i) gradients_xy[i] = gradients_x[i] * gradients_y[i];
    // 计算每一个点的加权之和, 先储存起来
    std::vector<double> xx_sum(length, 0), yy_sum(length, 0), xy_sum(length, 0);
    box_filter(gradients_xx.data(), xx_sum.data(), radius, H, W);
    box_filter(gradients_yy.data(), yy_sum.data(), radius, H, W);
    box_filter(gradients_xy.data(), xy_sum.data(), radius, H, W);
    // 开始计算每一个点的 harris 响应值
    std::vector<double> R(length, 0);
    const int H_radius = H - radius;
    const int W_radius = W - radius;
    for(int i = radius; i < H_radius; ++i) {
        double* const xx = xx_sum.data() + i * W;
        double* const yy = yy_sum.data() + i * W;
        double* const xy = xy_sum.data() + i * W;
        double* const res_ptr = R.data() + i * W;
        for(int j = radius; j < W_radius; ++j) {
            // 计算这个点所在窗口的加权和
            const double A = xx[j] / 255;
            const double B = yy[j] / 255;
            const double C = xy[j] / 255;
            // 计算 λ1 和 λ2
            const double det = A * B - C * C;
            const double trace = A + B;
            res_ptr[j] = det - alpha * (trace * trace);
        }
    }
    // 准备一个结果
    key_points_type detection;
    // 需要进行局部非极大化抑制
    for(int i = 1; i < H_1; ++i) {
        double* row_ptr = R.data() + i * W;
        for(int j = 1; j < W_1; ++j) {
            const double center = row_ptr[j];
            if(center > row_ptr[j - 1] and center > row_ptr[j + 1] and
               center > row_ptr[j - 1 - W] and center > row_ptr[j - W] and center > row_ptr[j + 1 - W] and
               center > row_ptr[j - 1 + W] and center > row_ptr[j + W] and center > row_ptr[j + 1 + W])
                if(center > threshold)
                    detection.emplace_back(center, i, j);
        }
    }
    // 取前 point_sum 个
    if(point_num > 0 and detection.size() > point_num) {
        // 按照响应值大小排序
        std::sort(detection.begin(), detection.end());
        std::reverse(detection.begin(), detection.end());
        detection.erase(detection.begin() + point_num, detection.end());
        detection.shrink_to_fit();
    }
    std::cout << "收集到  " << detection.size() << " 个角点 " << std::endl;
    return detection;
}








void demo_1() {
    std::string origin_path("../images/detail/harris_demo_1.png");
    const auto origin_image = cv::imread(origin_path);
    if(origin_image.empty()) {
        std::cout << "读取图像 " << origin_path << " 失败 !" << std::endl;
        return;
    }
    // 转成灰度图
    cv::Mat origin_gray;
    cv::cvtColor(origin_image, origin_gray, cv::COLOR_BGR2GRAY);

    // 检测 Harris 角点
    const auto harris_result = harris_corner_detect(origin_gray, 2, 0.04, 1e6, 0);

    // 画出来
    cv::Mat display = origin_image.clone();
    for(const auto& item : harris_result) {
        cv::circle(display, cv::Point(std::get<2>(item), std::get<1>(item)), 2, cv::Scalar(0, 0, 255), 3);
    }

    // 保存
    const auto comparison_results = cv_concat({display}, true);
    cv_show(comparison_results);

    // 做灰度变换, 
}





int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    // Laplace 检测边缘
    demo_1();

    return 0;
}
