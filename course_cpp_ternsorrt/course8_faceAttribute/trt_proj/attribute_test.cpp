#include <iostream>

#include "gflags/gflags.h"

#include "attribute.h"

DEFINE_string(model, "emotion.engine", "model path");
DEFINE_string(type, "gender", "attr type");
DEFINE_string(img, "", "image path");

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        // print error
        std::cout << "Usage: " << argv[0] << " --model=emotion.engine --type=gender --img=1.jpg" << std::endl;
        return -1;
    }

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::string model_path = FLAGS_model;
    std::string type = FLAGS_type;
    std::string img = FLAGS_img;

    attribute::Attribute att_model(model_path);
    cv::Mat frame = cv::imread(img);

    if (type == "gender")
        std::cout << att_model.run(frame, attribute::GENDER) << std::endl;
    else if (type == "age")
        std::cout << att_model.run(frame, attribute::AGE) << std::endl;
    else if (type == "mask")
        std::cout << att_model.run(frame, attribute::MASK) << std::endl;
    else if (type == "emotion")
        std::cout << att_model.run(frame, attribute::EMOTION) << std::endl;
    else
        std::cout << "type error" << std::endl;

    return 0;
}

