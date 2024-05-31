#pragma once
#include <opencv2/opencv.hpp>
#include "types.h"

cv::Rect getRect(const cv::Mat &img, const Detection& der, int orgH, int orgW);

void draw(cv::Mat& img, const std::vector<Detection>& ders, int orgH, int orgW);
