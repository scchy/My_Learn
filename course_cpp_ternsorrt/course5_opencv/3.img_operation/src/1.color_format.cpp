#include "opencv2/opencv.hpp"
#include <iostream>

int main()
{
    // 读取图片
    cv::Mat src = cv::imread("./media/dog.jpg");
    // BGR -> Gray
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    // BGR -> HSV Hue-色调， Saturation-饱和度 value-明度
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    // BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
    // 保存
    cv::imwrite("./output/1.gray.jpg", gray);
    cv::imwrite("./output/1.hsv.jpg", hsv);
    cv::imwrite("./output/1.rgb.jpg", rgb);
    
    return 0;
}