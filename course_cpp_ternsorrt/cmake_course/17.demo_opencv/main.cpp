#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "请输入图片路径" << std::endl;
        return -1;
    }
    else
    {
        std::cout << "图片路径为：" << argv[1] << std::endl;

        cv::Mat image = cv::imread(argv[1]);
        cv::Mat image_resized;
        cv::resize(image, image_resized, cv::Size(400, 400));
        cv::cvtColor(image_resized, image_resized, cv::COLOR_BGR2GRAY);
        cv::imwrite("resized.png", image_resized);

        std::cout << "图片已处理完毕，并保存为resized.png" << std::endl;
    }

    return 0;
}