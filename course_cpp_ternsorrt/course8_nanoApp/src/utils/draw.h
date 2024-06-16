#pragma once

#include <opencv2/opencv.hpp>

#include "types.h"

cv::Rect getRect(const cv::Mat &img, const Detection &det, int orgH, int orgW);

void draw(cv::Mat &img, const std::vector<Detection> &dets, int orgH, int orgW);

void draw(cv::Mat &img, float elapsed, const std::vector<DetWithAttr> &dets, int orgH, int orgW);

