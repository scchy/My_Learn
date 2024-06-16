#include "draw.h"

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

cv::Rect getRect(const cv::Mat &img, const Detection &det, int orgH, int orgW)
{

    float scalex = img.cols / (float)orgW;
    float scaley = img.rows / (float)orgH;

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

void draw(cv::Mat &img, const std::vector<Detection> &dets, int orgH, int orgW)
{
    for (auto &det : dets)
    {
        auto rect = getRect(img, det, orgH, orgW);
        cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);
        // cv::putText(img, std::to_string(int(det.class_id)), cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(img, std::to_string(det.conf), cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    }
}

void draw(cv::Mat &img, float elapsed, const std::vector<DetWithAttr> &dets, int orgH, int orgW)
{
    std::string fps_str = "fps: " + std::to_string(1000.f / elapsed);

    cv::putText(img, fps_str, cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 255, 255), 2);

    for (auto &det : dets)
    {
        auto rect = getRect(img, det.det, orgH, orgW);
        cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);

        cv::putText(img, det.name, cv::Point(rect.x, rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        // gender, age , mask , emotion
        cv::putText(img, det.gender, cv::Point(rect.x + rect.width + 10, rect.y + 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(img, det.age, cv::Point(rect.x + rect.width + 10, rect.y + 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(img, det.mask, cv::Point(rect.x + rect.width + 10, rect.y + 70), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(img, det.emotion, cv::Point(rect.x + rect.width + 10, rect.y + 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    }
}