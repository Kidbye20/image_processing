//C++
#include <cmath>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <iostream>
// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


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
    cv::Mat touint8(const std::vector<T>& source, const int H, const int W) {
        cv::Mat res(H, W, CV_8UC1);
        const int length = H * W;
        for(int i = 0;i < length; ++i) res.data[i] = std::abs(source[i]);
        return res;
    }

    cv::Mat get_rotated(const cv::Mat& source, const int angle, const cv::Size& _size, const cv::Point2f& center) {
        cv::Mat rotated_image;
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::warpAffine(source, rotated_image, rot_mat, _size, cv::INTER_LINEAR);
        return rotated_image;
    }

    cv::Mat my_rotate(const cv::Mat& source) {
        const int H = source.rows;
        const int W = source.cols;
        cv::Mat res(W, H, CV_8UC3);
        for(int i = 0;i < H; ++i) {
            for(int j = 0;j < W; ++j) {
                res.at<cv::Vec3b>(W - 1 - j, i)[0] = source.at<cv::Vec3b>(i, j)[0];
                res.at<cv::Vec3b>(W - 1 - j, i)[1] = source.at<cv::Vec3b>(i, j)[1];
                res.at<cv::Vec3b>(W - 1 - j, i)[2] = source.at<cv::Vec3b>(i, j)[2];
            }
        }
        return res;
    }
    // 代码取自 https://blog.csdn.net/qq_34784753/article/details/69379135
    double generateGaussianNoise(double mu, double sigma) {
        const double epsilon = std::numeric_limits<double>::min();
        static double z0, z1;
        static bool flag = false;
        flag = !flag;
        if (!flag) return z1 * sigma + mu;
        double u1, u2;
        do {
            u1 = std::rand() * (1.0 / RAND_MAX);
            u2 = std::rand() * (1.0 / RAND_MAX);
        } while (u1 <= epsilon);
        z0 = std::sqrt(-2.0 * log(u1)) * cos(2 * CV_PI * u2);
        z1 = std::sqrt(-2.0 * log(u1)) * sin(2 * CV_PI * u2);
        return z0 * sigma + mu;
    }
    // 为图像添加高斯噪声
    // 代码取自 https://blog.csdn.net/qq_34784753/article/details/69379135
    cv::Mat add_gaussian_noise(const cv::Mat &source) {
        cv::Mat res = source.clone();
        int channels = res.channels();
        int rows_number = res.rows;
        int cols_number = res.cols * channels;
        if (res.isContinuous()) {
            cols_number *= rows_number;
            rows_number = 1;
        }
        for (int i = 0; i < rows_number; i++) {
            for (int j = 0; j < cols_number; j++) {
                int val = res.ptr<uchar>(i)[j] +
                    generateGaussianNoise(2, 1.5) * 16;
                res.ptr<uchar>(i)[j] = cv::saturate_cast<uchar>(val);
            }
        }
        return res;
    }
}



using key_points_type = std::vector< std::pair<int, int> >;
key_points_type susan_corner_detect(const cv::Mat& source, const int radius, const int nms_radius=5, const int threshold=20) {
    // 获取图像信息
    const int H = source.rows;
    const int W = source.cols;
    const int length = H * W;
    cv::Mat gray_image;
    if(source.channels() == 3) cv::cvtColor(source, gray_image, cv::COLOR_BGR2GRAY);
    else gray_image = source;
    // 准备一个模板
    const int max_size = (2 * radius + 1) * (2 * radius + 1);
    std::vector<int> weights(max_size, 0);
    std::vector<int> offset(max_size, 0);
    const int radius_2 = (radius + 0.5) * (radius + 0.5);
    int cnt = 0;
    for(int i = -radius;i <= radius; ++i)
        for(int j = -radius; j <= radius; ++j)
            if(i * i + j * j <= radius_2) {
                weights[cnt] = 1;
                offset[cnt++] = i * W + j;
            }
    // 判断是否是同质点的阈值
    const int half_area = cnt / 2;
    // 存放每个点的响应值
    std::vector<int> response(length, 0);
    // 开始计算角点响应值
    const int H_radius = H - radius, W_radius = W - radius;
    std::cout << H_radius << " , " << W_radius << std::endl;
    for(int i = radius; i < H_radius; ++i) {
        uchar* const row_ptr = gray_image.data + i * W;
        int* const res_ptr = response.data() + i * W;
        for(int j = radius; j < W_radius; ++j) {
            // 首先, 判断这个圆形窗口和中心点的差小于阈值的
            const uchar center = row_ptr[j];
            double number = 0;
            for(int k = 0;k < cnt; ++k)
                if(std::abs(row_ptr[j + offset[k]] - center) < threshold)
                    number += weights[k];
            // 把中心点减去
            --number;
            // 如果同质点的加权之和小于一半的面积
            if(number < half_area)
                res_ptr[j] = half_area - number;
        }
    }
    cv_show(20 * touint8(response, H, W));
    cv_write(30 * touint8(response, H, W), "./images/output/corner_detection/SUSAN_response.png");
    // 准备结果
    key_points_type detection;
    // 非极大值抑制
    for(int i = nms_radius; i < H - nms_radius; ++i) {
        const int* const row_ptr = response.data() + i * W;
        for(int j = nms_radius; j < W - nms_radius; ++j) {
            // 找局部区域的所有点判断是否是极大值
            const int center = row_ptr[j];
            // center = 0 的点很可能就是平坦区域, 不参与比较
            if(center > 1) {
                bool flag = true;
                for(int x = -nms_radius; x <= nms_radius; ++x) {
                    const int* const cur_ptr = row_ptr + j + x * W;
                    for(int y = -nms_radius; y <= nms_radius; ++y) {
                        if(cur_ptr[y] >= center) {
                            if(x == 0 and y == 0) continue;
                            flag = false; break;
                        }
                    }
                    if(!flag) break;
                }
                // 如果真的是极大值
                if(flag) detection.emplace_back(i, j);
            }
        }
    }
    return detection;
}



void corner_detect_demo() {
    const std::string save_dir("./images/output/corner_detection/");
    std::string origin_path("../images/corner/harris_demo_1.png");
    const auto origin_image = cv::imread(origin_path);
    if(origin_image.empty()) {
        std::cout << "读取图像 " << origin_path << " 失败 !" << std::endl;
        return;
    }
    // 写一个展示的函数
    auto corner_display = [](const cv::Mat& source, const key_points_type& detection, const std::string save_path, const int radius=3, const int thickness=4)
            -> void {
        cv::Mat display = source.clone();
        for(const auto& item : detection)
        cv::circle(display, cv::Point(item.second, item.first), radius, cv::Scalar(0, 255, 0), thickness);
        cv_show(display);
        cv_write(display, save_path);
    };
    cv::Mat another_image;

    // 朴素的 SUSAN
    auto detection = susan_corner_detect(origin_image, 3, 15, 30);
    corner_display(origin_image, detection, save_dir + "horse.png");

    another_image = cv::imread("../images/corner/corner_2.png");
    detection = susan_corner_detect(another_image, 3, 10, 45);
    corner_display(another_image, detection, save_dir + "table.png");

    another_image = cv::imread("../images/corner/corner_1.png");
    detection = susan_corner_detect(another_image, 3, 5, 10);
    corner_display(another_image, detection, save_dir + "toy.png");

    another_image = cv::imread("../images/corner/corner_3.png");
    detection = susan_corner_detect(another_image, 3, 4, 30);
    corner_display(another_image, detection, save_dir + "house.png", 2, 2);

    another_image = cv::imread("../images/corner/a0515-NKIM_MG_6602.png");
    detection = susan_corner_detect(another_image, 3, 8, 50);
    corner_display(another_image, detection, save_dir + "French.png", 2, 2);

    another_image = cv::imread("../images/corner/a0423-07-06-02-at-07h35m36-s_MG_1355.png");
    detection = susan_corner_detect(another_image, 3, 7, 45);
    corner_display(another_image, detection, save_dir + "gugong.png", 2, 2);

    another_image = cv::imread("../images/corner/a0367-IMG_0338.png");
    detection = susan_corner_detect(another_image, 3, 7, 60);
    corner_display(another_image, detection, save_dir + "city.png", 2, 2);

    another_image = cv::imread("../images/corner/a0516-IMG_4420.png");
    detection = susan_corner_detect(another_image, 3, 10, 45);
    corner_display(another_image, detection, save_dir + "car.png", 2, 2);

    // 如果是噪声的话
    const auto noisy_image = add_gaussian_noise(cv::imread("../images/corner/a0515-NKIM_MG_6602.png"));
    detection = susan_corner_detect(noisy_image, 3, 10, 45);
    corner_display(noisy_image, detection, save_dir + "noisy.png", 2, 2);

    // 如果是旋转的话
    const auto rotated_image = my_rotate(cv::imread("../images/corner/a0515-NKIM_MG_6602.png"));
    detection = susan_corner_detect(rotated_image, 3, 10, 45);
    corner_display(rotated_image, detection, save_dir + "rotated.png", 2, 2);
}





int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    // 角点检测
    corner_detect_demo();

    return 0;
}
