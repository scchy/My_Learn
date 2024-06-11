#include "attribute.h"
#include "utils/preprocess.h"
#include "utils/postprocess.h"

namespace attribute
{
    Attribute::Attribute(const std::string &model_path)
    {
        engine_.reset(new TrtEngine(model_path));
    }

    std::string Attribute::run(const cv::Mat &img, attr_type type){
        preprocess(img);
        doInference();
        return postprocess(type);
    }

    void Attribute::doInference(){
        engine_->doInference();
    }

    void Attribute::preprocess(const cv::Mat &img){
        emotionPreprocess(
            img, kInputW, kInputH,
            engine_->getHostBuffer(kInputTensorName)
        );
    }

    std::string Attribute::postprocess(attr_type type){
        float *output = static_cast<float *>(engine_->getHostBuffer(kOutClass));
        // find the max value
        int max_index = 0;
        float max_value = 0;
        for(int i = 0; i < kOutputSize; i++){
            // std::cout << output[i] << std::endl;
            if(output[i] > max_value){
                max_index = i;
                max_value = output[i];
            }
        }

        switch (type)
        {
        case GENDER:
            return kAttribute_gender[max_index];
        case AGE:
            return kAttribute_age[max_index];
        case MASK:
            return kAttribute_mask[max_index];
        case EMOTION:
            return kAttribute_emotion[max_index];
        }
    }
}
