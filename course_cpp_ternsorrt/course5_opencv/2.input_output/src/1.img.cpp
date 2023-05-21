#include <iostream>
#include "opencv2/opencv.hpp"

int main(int argc, char** argv){
    // 读取图片
    cv::Mat img = cv::imread("./media/cat.jpg");
    // 判断是否读取成功
    if(img.empty()){
        std::cout << "无法读取图片" << std::endl;
        return 1;
    }
    // 大于图片高度和宽度
    std::cout << "图片高度: " << img.rows << " 宽度: " << img.cols  << std::endl;
    // 打印图片数据
    // std::cout << "图片data:" << cv::format(img, cv::Formatter::FMT_NUMPY) << std::endl;

    // 创建简单的灰度图片
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::imwrite("./output/gray.jpg", gray);
    // 显示 - 远程链接无法显示
    // cv::imshow("图片", img);
    // // 等待按键
    // cv::waitKey(0);
    return 0;
}



