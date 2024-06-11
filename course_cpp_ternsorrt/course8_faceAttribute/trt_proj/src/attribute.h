#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "engine.h"
#include "utils/types.h"

namespace attribute
{
    const int kInputH = 48;
    const int kInputW = 48;
    const int kinputC = 1;
    const int kOutputSize = 6;
    const char *kInputTensorName = "conv2d_input";
    const char *kOutClass = "dense_2";

    enum attr_type
    {
        GENDER,
        AGE,
        MASK,
        EMOTION
    };

    // gender
    const std::vector<std::string> kAttribute_gender = {"man", "woman"};
    // age
    const std::vector<std::string>
        kAttribute_age = {"0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80"};
    // emotion
    const std::vector<std::string> kAttribute_emotion = {"Angry", "Fear", "happy", "Neutral", "Sad", "Suprise"};
    // mask
    const std::vector<std::string> kAttribute_mask = {"mask", "nomask"};

    class Attribute
    {
    public:
        Attribute(const std::string &model_path);
        std::string run(const cv::Mat &img, attr_type type);
        ~Attribute() = default;
    private:
        void doInference();
        void preprocess(const cv::Mat &img);
        std::string postprocess(attr_type type);

        std::shared_ptr<TrtEngine> engine_;
    };

}