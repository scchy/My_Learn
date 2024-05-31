#include "draw.h"

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

cv::Rect getRect(const cv::Mat &img, const Detection& det, int orgH, int orgW){
    float scalex = img.cols / (float) orgW;
    float scaley = img.rows / (float) orgH;

    float x1 = det.bbox[0] * scalex;
    float y1 = det.bbox[1] * scaley;
    float x2 = det.bbox[2] * scalex;
    float y2 = det.bbox[3] * scaley;

    x1 = clamp(x1, 0, img.cols);
    y1 = clamp(y1, 0, img.rows);
    x2 = clamp(x2, 0, img.cols);
    y2 = clamp(y2, 0, img.rows);

    auto left = x1;
    auto width = clamp(x2 - x1, 0, img.cols);
    auto top = y1;
    auto height = clamp(y2 - y1, 0, img.rows);
    return cv::Rect(left, top, width, height);
}


void draw(cv::Mat& img, const std::vector<Detection>& dets, int orgH, int orgW){
    for(auto& det: dets){
        auto rect = getRect(img, det, orgH, orgW);
        cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);
        cv::putText(
            img, 
            std::to_string(det.conf), 
            cv::Point(rect.x, rect.y), 
            cv::FONT_HERSHEY_SIMPLEX, 1,
            cv::Scalar(0, 255, 0), 2
        );
    }
}
