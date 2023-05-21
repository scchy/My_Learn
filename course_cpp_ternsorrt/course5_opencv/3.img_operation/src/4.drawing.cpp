// 绘制文字和图形
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

int main()
{
    // 创建一个黑色图片，
    // 参数 大小、图像类型  CV_8UC3表示8位无符号整数，3通道
    cv::Mat img = cv::Mat::zeros(cv::Size(600, 600), CV_8UC3);

    // 绘制直线，绘制分别是图像 起点、终点、颜色、线宽、线类型
    cv::line(
        img, cv::Point(50, 50), cv::Point(350, 250),
        cv::Scalar(0, 0, 255), 2, cv::LINE_AA
        );
    // 绘制矩形，参数分别是图像、左上角、右下角、颜色、线宽、线型
    cv::rectangle(
      img,
      cv::Point(50, 50), cv::Point(350, 250),
      cv::Scalar(0, 255, 0), 2, cv::LINE_AA
    );
    // 绘制圆形，参数分别是图像、圆心、半径、颜色、线宽、线型
    cv::circle(
        img,
        cv::Point(200, 150), 100,
        cv::Scalar(255, 0, 0), 2, cv::LINE_AA
    );
    // 实心
    cv::circle(
        img, cv::Point(200, 150), 50, 
        cv::Scalar(255, 0, 0), -1, cv::LINE_AA
    );
    // ================== 使用vector绘制多边形 ==================
    std::vector<cv::Point> points_v;
    for(int i=0; i < 5; i++){
        points_v.push_back(cv::Point(rand() % 600, rand() % 600));
    }
    // 绘制多边形
    // 参数 图像，顶点容器，是否闭合, 颜色、线宽、线型
    cv::polylines(
        img,
        points_v, true,
        cv::Scalar(255, 0, 0), 2, 8, 0
    );
    // ================== 绘制文字 ==================
    // 参数分别是图像、文字、文字位置、字体、字体大小、颜色、线宽、线型
    cv::putText(
        img,
        "Hello World!", cv::Point(400, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0,
        cv::Scalar(255, 255, 255), 2, 8, 0
    );
    // 保存
    cv::imwrite("./output/4.drawing.jpg", img);
    return 0;
}