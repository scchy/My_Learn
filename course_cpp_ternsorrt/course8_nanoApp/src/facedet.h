#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <engine.h>
#include "utils/types.h"

namespace facedet
{
    const int kInputH = 416;
    const int kInputW = 736;
    const int kInputC = 3;
    const int kOutputSize = 46 * 26;
  
    class Facedet
    {
    public:
        Facedet(const std::string &model_path);
        ~Facedet() = default;
        std::vector<Detection> run(const cv::Mat &img);

    private:
        void doInference();
        void preprocess(const cv::Mat &img);
        std::vector<Detection> postprocess();

        std::shared_ptr<TrtEngine> engine_;
    };

}

