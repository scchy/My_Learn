#include "preprocess.h"


void facedetPreprocess(const cv::Mat& img, int inputW, int inputH, void* buffer)
{
    cv::Mat img_float;
    img.convertTo(img_float, CV_32FC3, 1.0/255.0);  // [0, 1]

    cv::Mat img_resized;
    cv::resize(img_float, img_resized, cv::Size(inputW, inputH));

    cv::Mat img_rgb;
    cv::cvtColor(img_resized, img_rgb, cv::COLOR_BGR2RGB);

    // convert to NCHW
    std::vector<cv::Mat> nchw_channels;
    cv::split(img_rgb, nchw_channels);
    for(auto &img: nchw_channels)
    {
        img = img.reshape(1, 1);
    }

    // rrrrr、ggggg、bbbbb拼接成rrrgggbbb
    cv::Mat nchw;
    cv::hconcat(nchw_channels, nchw);
    // to GPU
    memcpy(buffer, nchw.data, 3 * inputH * inputW * sizeof(float));
}


void facenetPreprocess(const cv::Mat &img, int inputW, int inputH, void *buffer)
{
    cv::Mat img_float;
    img.convertTo(img_float, CV_32FC3);
    cv::Mat img_resized;
    cv::resize(img_float, img_resized, cv::Size(inputW, inputH));

    cv::Mat img_rgb;
    cv::cvtColor(img_resized, img_rgb, cv::COLOR_BGR2RGB);

    cv::Mat img_normalized;
    cv::subtract(img_rgb, cv::Scalar(127.5, 127.5, 127.5), img_normalized);
    cv::divide(img_normalized, cv::Scalar(128.0, 128.0, 128.0), img_normalized);

    std::vector<cv::Mat> nchw_channels;
    // 将输入图像分解成三个单通道图像：rrrrr、ggggg、bbbbb
    cv::split(img_normalized, nchw_channels); 
    // 将每个单通道图像进行reshape操作，变为1x1xHxW的四维矩阵
    for (auto &img : nchw_channels)
    {
        // reshape参数分别是cn：通道数，rows：行数
        // 类似[[r,r,r,r,r]]或[[g,g,g,g,g]]或[[b,b,b,b,b]]，每个有width * height个元素
        img = img.reshape(1, 1);
    }
    // 将三个单通道图像拼接成一个三通道图像，即rrrrr、ggggg、bbbbb拼接成rrrgggbbb
    cv::Mat nchw;
    cv::hconcat(nchw_channels, nchw);
    // 将处理后的图片数据拷贝到GPU

    memcpy(buffer, nchw.data, 3 * inputH * inputW * sizeof(float));
}

