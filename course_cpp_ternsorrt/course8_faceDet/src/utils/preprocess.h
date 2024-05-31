#pragma once 
#include <opencv2/opencv.hpp>

void facenetPreprocess(const cv::Mat& img, int inputW, int inputH, void* buffer);
void facedetPreprocess(const cv::Mat& img, int inputW, int inputH, void* buffer);
void emotionPreprocess(const cv::Mat& img, int inputW, int inputH, void* buffer);

