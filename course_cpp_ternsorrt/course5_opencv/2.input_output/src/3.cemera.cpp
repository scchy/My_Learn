#include <opencv2/opencv.hpp>
#include <iostream>
#include <gflags/gflags.h>

DEFINE_int32(camera, 4, "Input camera"); // 摄像头编号

int main(int argc, char **argv){
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // 读取视频： 创建一个videoCapture对象，参数为摄像头编码
    cv::VideoCapture capture(FLAGS_camera);
    // 设置指定摄像头的分辨率
    int width = 640;
    int height = 480;

    // 设置摄像头的宽度高度
    capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    // 判断视频是否读取成功，
    if(!capture.isOpened())
    {
        std::cout << "无法打开摄像头: " << FLAGS_camera << std::endl;
        return 1;
    }
    // 读取视频帧，使用mat类型的frame存储返回的帧
    cv::Mat frame;
    // 写入MP4文件， 参数分别为：文件名，编码格式，帧率，帧大小
    cv::VideoWriter writer(
        "./output/record.mp4",
        cv::VideoWriter::fourcc('H', '2', '6', '4'),
        20,
        cv::Size(width, height)
    );
    // 循环读取视频
    while(true){
        // 读取视频帧，使用 >> 运算符或 read() 函数
        capture.read(frame);
        // capture >> frame;
        // flip
        cv::imshow("opencv demo", frame);
        // 写入视频
        writer.write(frame);
        // 等待按键，延迟30ms. 否则视频过快
        int k = cv::waitKey(30);
        if(k==27){
            std::cout << "退出" << std::endl;
            break;
        }
    }
    return 0;
}

