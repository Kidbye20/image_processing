// C++
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
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



class PoissonMatting {
public:
    PoissonMatting();
    void open(std::string filename);
    cv::Mat &getImageMat();
    void matting(cv::InputArray _trimap, cv::OutputArray _foreground, cv::OutputArray _alpha);
protected:
    cv::Mat img;
    static std::vector<cv::Point> findBoundaryPixels(const cv::Mat_<uchar> &trimap, int a, int b);
    void _matting(cv::Mat _image, cv::Mat _trimap, cv::Mat &_foreground, cv::Mat &_alpha);
};


template <class T>
T sqr(T t) {
    return t * t;
}

double dist_sqr(cv::Point p1, cv::Point p2) {
    return sqr(p1.x - p2.x) + sqr(p1.y - p2.y);
}
int color_dis(cv::Vec3b p1, cv::Vec3b p2) {
    int t1 = fmax(fmax(p1[0], p1[1]), p1[2]);
    int t2 = fmax(fmax(p2[0], p2[1]), p2[2]);
    return t1 - t2;
}

template <class T>
int inX(const T &image, int x) {
    if (x < 0) x = 0;
    if (x >= image.cols) x = image.cols - 1;
    return x;
}

template <class T>
int inY(const T &image, int y) {
    if (y < 0) y = 0;
    if (y >= image.rows) y = image.rows - 1;
    return y;
}

double intensity(cv::Vec3b v) {
    return fmax(fmax(v[0], v[1]), v[2]);
}

/***************************************************/

PoissonMatting::PoissonMatting()
{

}

void PoissonMatting::open(std::string filename)
{
    img = cv::imread(filename);
}

cv::Mat &PoissonMatting::getImageMat()
{
    return img;
}

void PoissonMatting::matting(cv::InputArray _trimap, cv::OutputArray _foreground, cv::OutputArray _alpha)
{
    cv::Mat image = img;
    cv::Mat trimap = _trimap.getMat();

    cv::Mat &foreground = _foreground.getMatRef();
    cv::Mat &alpha = _alpha.getMatRef();

    _matting(image, trimap, foreground, alpha);
}

std::vector<cv::Point> PoissonMatting::findBoundaryPixels(const cv::Mat_<uchar> &trimap, int a, int b)
{
    std::vector<cv::Point> result;

    for (int x = 1; x < trimap.cols - 1; ++x) {
        for (int y = 1; y < trimap.rows - 1; ++y) {
            if (trimap(y, x) == a) {
                if (trimap(y - 1, x) == b ||
                        trimap(y + 1, x) == b ||
                        trimap(y, x - 1) == b ||
                        trimap(y, x + 1) == b) {
                    result.push_back(cv::Point(x, y));
                }
            }
        }
    }

    return result;
}


void PoissonMatting::_matting(cv::Mat _image, cv::Mat _trimap, cv::Mat &_foreground, cv::Mat &_alpha)
{
    const cv::Mat_<cv::Vec3b> &image = static_cast<const cv::Mat_<cv::Vec3b> &>(_image);
    cv::Mat_<uchar> &trimap = static_cast<cv::Mat_<uchar> &>(_trimap);

    _foreground.create(image.size(), CV_8UC3);
    _alpha.create(image.size(), CV_8UC1);

    cv::Mat_<cv::Vec3b> &foreground = static_cast<cv::Mat_<cv::Vec3b>&>(_foreground);
    cv::Mat_<uchar> &alpha = static_cast<cv::Mat_<uchar>&>(_alpha);

    cv::Mat_<double> FminusB = cv::Mat_<double>::zeros(trimap.rows, trimap.cols);

    for (int times = 0; times < 5; ++times) {

        // Output the Progress
        std::vector<cv::Point> foregroundBoundary = findBoundaryPixels(trimap, 255, 128);
        std::vector<cv::Point> backgroundBoundary = findBoundaryPixels(trimap, 0, 128);

        cv::Mat_<uchar> trimap_blur;

        // Smooth Trimap by gaussian filter to denoise
        cv::GaussianBlur(trimap, trimap_blur, cv::Size(9, 9), 0);

        // Build the F-B Map
        for (int x = 0; x < trimap.cols; ++x) {
            for (int y = 0; y < trimap.rows; ++y) {
                cv::Point current;
                current.x = x;
                current.y = y;
                if (trimap_blur(y, x) == 255) {
                    FminusB(y, x) = color_dis(image(y, x), cv::Vec3b(0, 0, 0));
                } else if (trimap_blur(y, x) == 0) {
                    FminusB(y, x) = color_dis(cv::Vec3b(0, 0, 0), image(y, x));
                } else {
                    // is in Unknown Area
                    // Find Nearest Foreground and Background Point
                    cv::Point nearestForegroundPoint, nearestBackgroundPoint;
                    double nearestForegroundDistance = 1e9, nearestBackgroundDistance = 1e9;
                    for(cv::Point &p : foregroundBoundary) {
                        double t = dist_sqr(p, current);
                        if (t < nearestForegroundDistance) {
                            nearestForegroundDistance = t;
                            nearestForegroundPoint = p;
                        }
                    }
                    for(cv::Point &p : backgroundBoundary) {
                        double t = dist_sqr(p, current);
                        if (t < nearestBackgroundDistance) {
                            nearestBackgroundDistance = t;
                            nearestBackgroundPoint = p;
                        }
                    }
                    // Calculate F - B
                    FminusB(y, x) = color_dis(image(nearestForegroundPoint.y, nearestForegroundPoint.x),
                                              image(nearestBackgroundPoint.y, nearestBackgroundPoint.x));
                    if (FminusB(y, x) == 0)
                        FminusB(y, x) = 1e-9;
                }
            }
        }

        // Smooth (F - B) image by Gaussian filter
        cv::GaussianBlur(FminusB, FminusB, cv::Size(9, 9), 0);

        // Solve the Poisson Equation By The Gauss-Seidel Method (Iterative Method)
        for (int times2 = 0; times2 < 300; ++times2) {
            for (int x = 0; x < trimap.cols; ++x) {
                for (int y = 0; y < trimap.rows; ++y) {
                    if (trimap(y, x) == 128) {
                        // is in Unknown Area
#define I(x, y) (intensity(image(inY(image, y), inX(image, x))))
#define FmB(y, x) (FminusB(inY(FminusB, y), inX(FminusB, x)))
                        // Calculate the divergence
                        double dvgX = ((I(x + 1, y) + I(x - 1, y) - 2 * I(x, y)) * FmB(y, x)

                                - (I(x + 1, y) - I(x, y)) * (FmB(y, x + 1) - FmB(y, x)))
                                / (FmB(y, x) * FmB(y, x));
                        double dvgY = ((I(x, y + 1) + I(x, y - 1) - 2 * I(x, y)) * FmB(y, x)
                                - (I(x, y + 1) - I(x, y)) * (FmB(y + 1, x) - FmB(y, x)))
                                / (FmB(y, x) * FmB(y, x));
                        double dvg = dvgX + dvgY;
#undef FmB
#undef I
                        // Calculate the New Alpha (Gauss-Seidel Method)
                        double newAlpha = (((double)alpha(y, x + 1)
                                        + alpha(y, x - 1)
                                        + alpha(y + 1, x)
                                        + alpha(y - 1, x)
                                        - dvg * 255.0) / 4.0);
                        // Update the Trimap
                        if (newAlpha > 253) {
                            // fore
                            trimap(y, x) = 255;
                        } else if (newAlpha < 3) {
                            // back
                            trimap(y, x) = 0;
                        }
                        // Avoid overflow
                        if (newAlpha < 0) {
                            newAlpha = 0;
                        }
                        if (newAlpha > 255) {
                            newAlpha = 255;
                        }
                        // Assign new alpha
                        alpha(y, x) = newAlpha;
                    } else if (trimap(y, x) == 255) {
                        // is Foreground
                        alpha(y, x) = 255;
                    } else if (trimap(y, x) == 0) {
                        // is Background
                        alpha(y, x) = 0;
                    }
                }
            }
        }
    }

    // Generate Foreground Image (Red Background)
    for (int x = 0; x < alpha.cols; ++x) {
        for (int y = 0; y < alpha.rows; ++y) {
            foreground(y, x) = ((double) alpha(y, x) / 255) * image(y, x) + ((255.0 - alpha(y, x)) / 255 * cv::Vec3b(0, 0, 255));
        }
    }
}


int main() {

    // void matting(cv::InputArray _trimap, cv::OutputArray _foreground, cv::OutputArray _alpha);

    PoissonMatting solver;
    cv::Mat image = cv::imread("./images/input/mask_4.bmp");
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    solver.open("./images/input/input_4.bmp");
    cv::Mat foreground, alpha;
    solver.matting(image, foreground, alpha);
    cv_show(foreground);
    return 0;
}