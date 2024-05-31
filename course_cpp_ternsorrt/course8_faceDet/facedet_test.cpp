#include <iostream>
#include "gflags/gflags.h"
#include "facedet.h"
#include "utils/draw.h"

DEFINE_string(model, "facedet.engine", "facenet model path");
DEFINE_string(img, "", "image path");


int main(int argc, char **argv){
    if(argc < 2){
        //error
        std::cout << "Usage: ./facedet_test --model=xx.engine --img=xx.png" << std::endl;
    }

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::string model_path = FLAGS_model;
    std::string img = FLAGS_img;

    facedet::Facedet facedet(model_path);
    cv::Mat frame = cv::imread(img, cv::IMREAD_COLOR);

    std::cout << "Finished Reading " << img << "; frame.empty()=" << frame.empty() << std::endl;
    auto dets = facedet.run(frame);
    int i = 0;
    for(auto det: dets){
        // crop img and imwrite
        cv::Mat crop_img = frame(getRect(frame, det, facedet::kInputH, facedet::kInputW));
        // save all croped img
        cv::imwrite("imges/crop/crop_" + std::to_string(i) + ".jpg", crop_img);
        i++;
    }

    draw(frame, dets, facedet::kInputH, facedet::kInputW);
    cv::imwrite("out.png", frame);
    return 0;
}

