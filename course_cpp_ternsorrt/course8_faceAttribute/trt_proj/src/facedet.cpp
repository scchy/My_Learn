#include "facedet.h"
#include "utils/preprocess.h"
#include "utils/postprocess.h"
#include <vector>


namespace facedet 
{
    const char *kInputTensorName = "input_1";
    const char *kOutBBox = "output_bbox/BiasAdd";
    const char *kOutConf = "output_cov/Sigmoid";

    Facedet::Facedet(const std::string &model_path){
        engine_.reset(new TrtEngine(model_path)); // 作用：创建一个TrtEngine对象，然后将其指针赋值给engine_
    }

    std::vector<Detection> Facedet::run(const cv::Mat &img){
        preprocess(img);
        doInference();
        return postprocess();
    }

    void Facedet::doInference(){
        engine_ -> doInference();
    }

    void Facedet::preprocess(const cv::Mat &img){
        facedetPreprocess(img, kInputW, kInputH, engine_->getHostBuffer(kInputTensorName));
    }

    std::vector<Detection> Facedet::postprocess(){
        std::vector<float> bbox(kOutputSize * 4);
        std::vector<float> conf(kOutputSize);
        memcpy(bbox.data(), engine_->getHostBuffer(kOutBBox), 4 * kOutputSize * sizeof(float));
        memcpy(conf.data(), engine_->getHostBuffer(kOutConf), kOutputSize * sizeof(float));
        std::vector<Detection> detections;
        float gridWidth = kInputW / 46;
        float gridHeight = kInputH / 26;
        float bbox_norm = 35;
        float offset = 0.5;
        for(int i = 0; i < kOutputSize; ++i){
            int gridx = i % 46;
            int gridy = i / 46;
            float cx = float(gridx * gridWidth + offset) / bbox_norm;
            float cy = float(gridy * gridHeight + offset) / bbox_norm;
            if( conf[i] > 0.5 ){
                std::cout << conf[i] << std::endl;
                std::cout << bbox[i];
                std::cout << " " << bbox[i + 1 * kOutputSize];
                std::cout << " " << bbox[i + 2 * kOutputSize];
                std::cout << " " << bbox[i + 3 * kOutputSize];
                std::cout << std::endl;
                Detection det;
                float bw = bbox[i + 2 * kOutputSize] * 35;
                float bh = bbox[i + 3 * kOutputSize] * 35;
                det.bbox[0] = (bbox[i] - cx) * bbox_norm * -1;
                det.bbox[1] = (bbox[i + 1 * kOutputSize] - cy) * bbox_norm * -1;
                det.bbox[2] = (bbox[i + 2 * kOutputSize] + cx) * bbox_norm;
                det.bbox[3] = (bbox[i + 3 * kOutputSize] + cy) * bbox_norm;
                det.conf = conf[i];
                detections.push_back(det);
            }
        }
        return nms(detections, 0.45);
    }
}