// tensorrt code load facenet tensorrt model and do inference
#pragma once 
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "engine.h"
#include "facedet.h"

namespace facenet
{
    const int kInputH = 112;
    const int kInputW = 112;
    const int kInputC = 3;
    const int kOutputSize = 256;
    const char *kInputTensorName = "input";
    const char *kOutTensorName = "output";
    const float kThreshold = 1.1;

    class Facenet
    {
    public:
        Facenet(const std::string &model_path);
        void loadSavedFeatures(const std::string &saved_face_list, facedet::Facedet *facedet_model);
        ~Facenet() = default;
        std::string run(const cv::Mat &img);

    private:
        void doInference();
        void preprocess(const cv::Mat &img);
        std::string postprocess();

        std::shared_ptr<TrtEngine> engine_;
        std::vector<std::vector<float>> saved_features_;
        std::vector<std::string> saved_faces_;
    };

};

