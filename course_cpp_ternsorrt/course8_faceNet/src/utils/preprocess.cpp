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


