#include <iostream>
#include "gflags/gflags.h"
#include "facedet.h"
#include "facenet.h"
#include "utils/draw.h"


// detector
DEFINE_string(facedet, "weights/detect.engine", "facenet model path");
// face net 
DEFINE_string(facenet, "weights/facenet_sim.engine", "facenet model path");
DEFINE_string(img, "", "img path");
DEFINE_string(faces, "face_list.txt", "faces");


int main(int argc, char **argv)
{
    if(argc < 2){
        std::cout << "usage: ./facedet_test --facedet=weights/detect.engine --facenet=weights/facenet_sim.engine --img=imgs/1.jpg --faces=face_list.txt" << std::endl;
        return -1;
    }

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::string det_path = FLAGS_facedet;
    std::string face_path = FLAGS_facenet;
    std::string img = FLAGS_img;
    facedet::Facedet facedet_model(det_path);
    facenet::Facenet facenet(face_path);
    std::cout << "FLAGS_faces=" << FLAGS_faces << std::endl;
    facenet.loadSavedFeatures(FLAGS_faces, &facedet_model);

    auto frame = cv::imread(img);
    auto faces = facedet_model.run(frame);
    std::cout << "faces: " << faces.size() << std::endl;
    for(auto &face: faces){
        cv::Mat crop_face = frame(getRect(frame, face, facedet::kInputH, facedet::kInputW));
        // similar judge
        auto name = facenet.run(crop_face);
        std::cout << std::endl;
        std::cout << "name: " << name << std::endl;
    }
    return 0;

}

