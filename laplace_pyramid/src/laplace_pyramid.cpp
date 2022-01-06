// C++
#include <vector>
#include <chrono>
#include <fstream>
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


    cv::Mat cv_concat(const std::vector<cv::Mat> images, const bool v=false) {
        cv::Mat result;
        if(not v) cv::hconcat(images, result);
        else cv::vconcat(images, result);
        return result;
    }

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }

    template<typename T>
    cv::Mat toint8(const std::vector<T>& source, const int H, const int W, const int C, const int _type) {
        cv::Mat result(H, W, _type);
        const int length = H * W * C;
        for(int i = 0;i < length; ++i) result.data[i] = std::abs(source[i]);
        return result;
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

void pyramid_upsample_interpolate(const cv::Mat& source) {
    const int H = source.rows, W = source.cols;
    const int C = source.channels();
    // 先处理第一行
    for(int ch = 0; ch < C; ++ch) {
        for(int j = 1;j < W - 1; j += 2)
            source.data[j * C + ch] = (source.data[(j - 1) * C + ch] + source.data[(j + 1) * C + ch]) / 2;
        source.data[(W - 1) * C + ch] = source.data[(W - 2) * C + ch];
    }
    // 最后一行
    uchar* const row_ptr = source.data + (H - 1) * W * C;
    for(int ch = 0; ch < C; ++ch) {
        for(int j = 1;j < W - 1; j += 2)
            row_ptr[j * C + ch] = (row_ptr[(j - 1) * C + ch] + row_ptr[(j + 1) * C + ch]) / 2;
        row_ptr[(W - 1) * C + ch] = row_ptr[(W - 2) * C + ch];
    }
    // 第一列
    for(int ch = 0; ch < C; ++ch) {
        for(int i = 1;i < H - 1; i += 2) {
            const int pos = i * W * C + ch;
            source.data[pos] = (source.data[pos - W * C] + source.data[pos + W * C]) / 2;
        }
    }
    // 剩下的行和列
    for(int ch = 0; ch < C; ++ch) {
        for(int i = 1;i < H - 1; ++i) {
            uchar* const row_ptr = source.data + i * W * C;
            for(int j = 1;j < W - 1; ++j) {
                // 如果都是奇数, 说明是空的
                if((i & 1) and (j & 1)) {
                    row_ptr[j * C + ch] = (row_ptr[(j - 1 - W) * C + ch] + row_ptr[(j - 1 + W) * C + ch] + row_ptr[(j + 1 - W) * C + ch] + row_ptr[(j + 1 + W) * C + ch]) / 4;
                }
                // 如果奇数行, 偶数列
                else if(i & 1) {
                    row_ptr[j * C + ch] = (row_ptr[(j - W) * C + ch] + row_ptr[(j + W) * C + ch]) / 2;
                }
                // 如果偶数行, 奇数列
                else if(j & 1) {
                    row_ptr[j * C + ch] = (row_ptr[(j - 1) * C + ch] + row_ptr[(j + 1) * C + ch]) / 2;
                }
            }
        }
    }
    // 最后一列是全黑的 ! 因为之前 upsample 的时候偶数列全部都是 0, 直接把倒数第二列拷贝过去
    // 而倒数第二列在前面的大循环中才会赋值, 所以放到这里
    for(int ch = 0; ch < C; ++ch) {
        for(int i = 1;i < H - 1; ++i) {
            const int pos = (i * W + W - 1) * C + ch;
            source.data[pos] = source.data[pos - C];
        }
    }
}



std::vector< std::vector<int> > build_laplace_pyramid(const std::vector<cv::Mat>& gaussi_pyramid) {
    // 查看几层
    const int layers_num = gaussi_pyramid.size();
    // 准备一个结果
    std::vector< std::vector<int> > laplace_pyramid;
    laplace_pyramid.reserve(layers_num - 1);
    // 从低分辨率开始构建拉普拉斯金字塔
    for(int i = layers_num - 1; i >= 1; --i) {
        // 首先低分辨率先上采样到两倍大小
        cv::Mat upsampled = pyramid_upsample(gaussi_pyramid[i]);
        // 填补值
        pyramid_upsample_interpolate(upsampled);
        // 放到拉普拉斯金字塔
        const int length = upsampled.rows * upsampled.cols * upsampled.channels();
        std::vector<int> residual(length, 0);
        for(int k = 0;k < length; ++k)
            residual[k] = gaussi_pyramid[i - 1].data[k] - upsampled.data[k];
        laplace_pyramid.emplace_back(residual);
    }
    std::reverse(laplace_pyramid.begin(), laplace_pyramid.end());
    return laplace_pyramid;
}



cv::Mat rebuild_image_from_laplace_pyramid(
        const cv::Mat& low_res,
        const std::vector< std::vector<int> >& laplace_pyramid,
        const int layers_num,
        const std::vector<cv::Mat>& gaussi_pyramid={}) {
    // 从最低分辨率的开始
    cv::Mat reconstructed = low_res.clone();
    for(int i = layers_num - 2;i >= 0; --i) {
        cv::Mat upsampled = pyramid_upsample(reconstructed);
        pyramid_upsample_interpolate(upsampled);
        // 将 laplace_pyramid 和 当前结果结合
        const int H = upsampled.rows, W = upsampled.cols, C = upsampled.channels();
        const int length = H * W * C;
        std::vector<uchar> temp(length, 0);
        for(int k = 0;k < length; ++k)
            temp[k] = cv::saturate_cast<uchar>(upsampled.data[k] + laplace_pyramid[i][k]);
        // 再把结果拷贝到 reconstructed
        reconstructed = cv::Mat(H, W, upsampled.type()); // reconstructed 的大小得变化一下
        std::memcpy(reconstructed.data, temp.data(), length);
        // 观察损失有多大
        if(gaussi_pyramid.size() == layers_num) {
            std::cout << "PSNR ===>  " << cv::PSNR(reconstructed, gaussi_pyramid[i]) << "db" << std::endl;
            cv_show(cv_concat({
                gaussi_pyramid[i],
                upsampled,
                toint8(laplace_pyramid[i], H, W, C, upsampled.type()),
                reconstructed}));
        }
    }
    return reconstructed;
}


void laplace_decomposition_demo() {
    // 读取图像
    const std::string image_path("./images/input/a2376-IMG_2891.png");
    const std::string save_dir("./images/output/");
    cv::Mat origin_image = cv::imread(image_path);
    assert(!origin_image.empty() and "图片读取失败");
    // 构建层数
    const int layers_num = 5;
    // 根据图像构建高斯金字塔
    const auto gaussi_pyramid = build_gaussi_pyramid(origin_image, layers_num);
    // 构建拉普拉斯金字塔
    auto laplace_pyramid = build_laplace_pyramid(gaussi_pyramid);
    // 展示
    cv::Mat reconstructed = rebuild_image_from_laplace_pyramid(
        gaussi_pyramid[layers_num - 1],
        laplace_pyramid,
        layers_num
        , gaussi_pyramid
    );

    // 从上面可以看出, 如何压缩信息 ?
    // 最小分辨率的图像 + 拉普拉斯金字塔
    // 而拉普拉斯金字塔很多都是 0, 所以只需要保留那些大于某个阈值的点的坐标信息, 即可大概还原出图像来
    // 拉普拉斯金字塔压缩信息(base, detail 分离, 弱化一些不必要的 detail)

    // 对拉普拉斯金字塔的结果过滤一些不必要的细节
    const int threshold = 10;
    std::vector< std::vector< std::vector<int> > > laplace_pyramid_compressd;
    laplace_pyramid_compressd.reserve(layers_num - 1);
    const int C = origin_image.channels();
    for(auto& layer : laplace_pyramid) {
        std::vector< std::vector<int> > this_layer;
        const int length = layer.size() / 3;
        for(int i = 0;i < length; ++i) {
            const int pos = C * i;
            int res_sum = 0.0;
            for(int k = 0;k < C; ++k)
                res_sum += std::abs(layer[pos + k]);
            // 判断这个点的细节强弱程度
            if(res_sum > threshold) {
                std::vector<int> temp(1 + C);
                temp[0] = i;
                for(int k = 0;k < C; ++k) temp[1 + k] = layer[pos + k];
                this_layer.emplace_back(temp);
            }
            else for(int k = 0;k < C; ++k) layer[pos + k] = 0;
        }
        std::cout << this_layer.size() << "/" << length << std::endl;
        laplace_pyramid_compressd.emplace_back(this_layer);
    }
    std::reverse(laplace_pyramid_compressd.begin(), laplace_pyramid_compressd.end());
    // 重新构建一次
    cv::Mat compressed = rebuild_image_from_laplace_pyramid(
        gaussi_pyramid[layers_num - 1],
        laplace_pyramid,
        layers_num
        , gaussi_pyramid
    );

    // 压缩前后都写成二进制文件存储, 看下都占据多大存储, 然后测一下重建时间

    // 如果是原图的话, 只需要遍历
    std::ofstream writer_1("./images/output/reconstructed", std::ios::binary);
    int H_1 = reconstructed.rows;
    int W_1 = reconstructed.cols;
    int C_1 = reconstructed.channels();
    const int length_1 = sizeof(uchar) * H_1 * W_1 * C_1;
    int info[4] = {reconstructed.rows, reconstructed.cols, reconstructed.channels(), reconstructed.type()};
    writer_1.write(reinterpret_cast<const char *>(&info), sizeof(info));
    writer_1.write(reinterpret_cast<const char *>(&reconstructed.data[0]), static_cast<std::streamsize>(length_1));
    writer_1.close();

    // 读取这个二进制文件, 重建
    run([&](){
        std::ifstream reader_1("./images/output/reconstructed", std::ios::binary);
        int info_2[4];
        reader_1.read((char*)(&info_2), sizeof(info_2));
        cv::Mat reconstructed_from_file(info_2[0], info_2[1], info_2[3]); // 根据高宽生命图像
        reader_1.read((char*)(&reconstructed_from_file.data[0]), sizeof(uchar) * info_2[0] * info_2[1] * info_2[2]);
        reader_1.close();
        cv_show(reconstructed_from_file, std::to_string(cv::PSNR(reconstructed, reconstructed_from_file)).c_str());
    }, "从无信息损失的二进制文件中读取  "); // 算时间的话, 要把上面的 cv_show 注释掉

    // 如果是拉普拉斯金字塔压缩过后的图像, 只需要保存最小分辨率的图像和拉普拉斯金字塔

    // 保存最小分辨率的图像
    std::ofstream writer_2("./images/output/compressed", std::ios::binary);
    const cv::Mat& start = gaussi_pyramid[layers_num - 1];
    int info_3[4] = {start.rows, start.cols, start.channels(), start.type()};
    int length_2 = sizeof(uchar) * info_3[0] * info_3[1] * info_3[2];
    writer_2.write(reinterpret_cast<const char *>(&info_3), sizeof(info_3));
    writer_2.write(reinterpret_cast<const char *>(&start.data[0]), static_cast<std::streamsize>(length_2));
    // 现在开始保存压缩之后的拉普拉斯金字塔
    const int pixel_info = (1 + origin_image.channels());
    const int laplace_num = laplace_pyramid_compressd.size();
    writer_2.write(reinterpret_cast<const char *>(&laplace_num), sizeof(int));
    for(int i = 0;i < laplace_num; ++i) {
        const int pixel_num = laplace_pyramid_compressd[i].size();
        writer_2.write(reinterpret_cast<const char *>(&pixel_num), sizeof(int));
        for(int k = 0;k < pixel_num; ++k)
            writer_2.write(reinterpret_cast<const char *>(&(laplace_pyramid_compressd[i][k][0])), sizeof(int) * pixel_info);
    }
    writer_2.close();
    // 从压缩过后的信息中恢复原图
    run([&](){
                std::ifstream reader_2("./images/output/compressed", std::ios::binary);
        int info_4[4];
        reader_2.read((char*)(&info_4), sizeof(info_4));
        cv::Mat reconstructed_from_compressed_file(info_4[0], info_4[1], info_4[3]); // 根据高宽生命图像
        reader_2.read((char*)(&reconstructed_from_compressed_file.data[0]), sizeof(uchar) * info_4[0] * info_4[1] * info_4[2]);
        cv_show(reconstructed_from_compressed_file);
        // 开始重新打造拉普拉斯金字塔
        int cur_H = info_4[0], cur_W = info_4[1], cur_C = info_4[2];
        std::vector< std::vector<int> > laplace_pyramid_reconstructed;
        int laplace_num_2;
        reader_2.read((char*)(&laplace_num_2), sizeof(int));
        laplace_pyramid_reconstructed.reserve(laplace_num_2);
        for(int i = 0;i < laplace_num_2; ++i) {
            cur_H *= 2, cur_W *= 2;
            std::vector<int> cur_laplace(cur_H * cur_W * cur_C, 0);
            int pixel_num;
            reader_2.read((char*)(&pixel_num), sizeof(int));
            for(int p = 0;p < pixel_num; ++p) {
                // 在 laplace 中找到对应的位置
                int pos;
                reader_2.read((char*)(&pos), sizeof(int));
                int* pos_c = cur_laplace.data() + pos * cur_C;
                for(int k = 0;k < cur_C; ++k)
                    reader_2.read((char*)(pos_c + k), sizeof(int));
            }
            laplace_pyramid_reconstructed.emplace_back(cur_laplace);
        }
        std::reverse(laplace_pyramid_reconstructed.begin(), laplace_pyramid_reconstructed.end());
        reader_2.close();

        cv::Mat reconstructed_from_compressed = rebuild_image_from_laplace_pyramid(
            gaussi_pyramid[layers_num - 1],
            laplace_pyramid_reconstructed,
            laplace_num_2 + 1
            , gaussi_pyramid
        );
    }, "根据拉普拉斯金字塔压缩的方法, 恢复图像 ");
}




int main() {

    // 拉普拉斯金字塔分解
    laplace_decomposition_demo();



    return 0;
}
