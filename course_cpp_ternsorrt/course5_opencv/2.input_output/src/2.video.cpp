#include <opencv2/opencv.hpp>
#include <iostream>
#include <gflags/gflags.h>

DEFINE_string(video, "./media/dog.mp4", "Inpute Vedio");

int main(int argc, char **argv){
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    cv::VideoCapture capture(FLAGS_video);
    if( !capture.isOpened()){
        std::cout << "无法读取视频：" << FLAGS_video << std::endl;
    }
    cv::Mat frame;
    cv::Mat gray_frame;
    while(true){
        // 读取视频，并赋值给frame
        capture.read(frame);
        // 判断是否读取成功
        if(frame.empty()){
            std::cout << "文件读取完毕" << std::endl;
            break;
        }
        // 转成灰度图
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        // 显示视频帧
        cv::imshow("raw frame:", frame);
        cv::imshow("gray frame:", gray_frame);
        int k = cv::waitKey(30);
        // 按下esc建退出
        if(k == 27){
            std::cout << "退出" << std::endl;
            break;
        }

    }
    return 0;
}