#include "opencv2/opencv.hpp"
#include <iostream>

int main()
{
    cv::Mat src = cv::imread("./media/dog.jpg");
    // 高斯模糊
    cv::Mat blur;
    // 三个参数 输入图像，输出图像，卷积核大小
    cv::GaussianBlur(src, blur, cv::Size(7, 7), 0);

    // 膨胀
    cv::Mat dilate;
    cv::dilate(
        src, dilate, 
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5))
    );

    // 腐蚀
    cv::Mat erode;
    cv::erode(
        src, 
        erode, 
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5))
    );

    // 保存
    cv::imwrite("./output/2.blur.jpg", blur);
    cv::imwrite("./output/2.dilate.jpg", dilate);
    cv::imwrite("./output/2.erode.jpg", erode);
    return 0;
}