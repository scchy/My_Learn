#include "facenet.h"
#include "utils/draw.h"
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "utils/preprocess.h"
#include "facedet.h"


namespace facenet
{
    Facenet::Facenet(const std::string &model_path){
        engine_.reset(new TrtEngine(model_path));
    }
    
    void Facenet::loadSavedFeatures(const std::string &saved_face_list, facedet::Facedet *facedet_model){
        std::ifstream ifs(saved_face_list);
        assert(ifs.is_open() && "Unable to load saved face file.");

        std::string line;
        while(std::getline(ifs, line))
        {
            std::cout << "filename: " << line << std::endl;
            // get the folder name "zhonghanliang" in format of crop/zhonghanliang/7.jpg

            size_t pos2 = line.find_last_of('/');
            size_t pos1 = line.find_last_of('/', pos2 - 1);
            std::string person_label = line.substr(pos1 + 1, pos2 - pos1 - 1);

            cv::Mat img = cv::imread(line);
            // detect one face
            auto faces = facedet_model->run(img);
            if (faces.size() == 0)
            {
                std::cout << "No face detected in " << line << std::endl;
                continue;
            }
            else
            {
                // 人脸检测，只取第一个人脸
                auto face = faces[0];
                // 人脸裁剪
                cv::Mat crop_face = img(
                    getRect(img, face, facedet::kInputH, facedet::kInputW)
                ); // 裁剪人脸
                // 人脸特征提取
                preprocess(crop_face);
                doInference();
                std::vector<float> feature(kOutputSize);
                memcpy(
                    feature.data(), engine_->getHostBuffer(kOutTensorName),
                    kOutputSize * sizeof(float)
                );
                saved_features_.push_back(feature);
                // 添加人脸名字
                saved_faces_.push_back(person_label);
                std::cout << "Load face feature of " << person_label << ", file: " << line << std::endl;
            }
        }
    }

    std::string Facenet::run(const cv::Mat &img)
    {
        preprocess(img);
        doInference();
        return postprocess();
    }
    
    void Facenet::doInference()
    {
        engine_->doInference(1);
    }
    
    void Facenet::preprocess(const cv::Mat &img)
    {
        facenetPreprocess(img, kInputH, kInputW, engine_->getHostBuffer(kInputTensorName));
    }
    // 设定距离阈值，计算欧式距离，选择最小的欧式距离
    std::string Facenet::postprocess()
    {
        std::vector<float> feature(kOutputSize);
        std::string unknown("uknown");
        memcpy(feature.data(), engine_->getHostBuffer(kOutTensorName), kOutputSize * sizeof(float));
        // 遍历所有已知人脸特征，计算欧式距离
        float min_dist = 1000.0;
        int min_index = -1;
        for (int i = 0; i < saved_features_.size(); ++i)
        {
            float dist = 0.0;
            //  计算欧式距离
            for (int j = 0; j < kOutputSize; ++j)
            {
                // 计算欧式距离
                dist += std::pow((feature[j] - saved_features_[i][j]), 2);
            }
            dist = std::sqrt(dist);
            // std::cout << dist << " " << saved_faces_[i] << std::endl;
            if (dist < min_dist)
            {
                min_dist = dist;
                min_index = i;
            }
        }
        if (min_dist < kThreshold)
        {
            return saved_faces_[min_index];
        }
        return unknown;
    }
}