#include <iostream>
#include "gflags/gflags.h"

#include "facedet.h"
#include "facenet.h"
#include "attribute.h"
#include "utils/draw.h"
#include "streamer/streamer.hpp"

DEFINE_string(facedet, "weights/detect.engine", "facenet model path");      // 人脸检测模型
DEFINE_string(facenet, "weights/facenet_sim.engine", "facenet model path"); // 人脸识别模型

DEFINE_string(att_gender, "weights/gender_sim.engine", "facenet model path");   // 性别识别模型
DEFINE_string(att_age, "weights/age_sim.engine", "facenet model path");         // 年龄识别模型
DEFINE_string(att_emotion, "weights/emotion_sim.engine", "facenet model path"); // 表情识别模型
DEFINE_string(att_mask, "weights/mask_sim.engine", "facenet model path");       // 口罩识别模型

DEFINE_string(vid, "", "video path");
DEFINE_string(faces, "face_list.txt", "faces");


int main(int argc, char **argv)
{
    if (argc < 2)
    {
        // print usage
        std::cout << "usage: ./stream --vid=video_path" << std::endl;
        return -1;
    }

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // 获取命令行参数
    std::string det_path = FLAGS_facedet;
    std::string face_path = FLAGS_facenet;

    std::string gender_path = FLAGS_att_gender;
    std::string age_path = FLAGS_att_age;
    std::string emotion_path = FLAGS_att_emotion;
    std::string mask_path = FLAGS_att_mask;

    std::string vid = FLAGS_vid;

    facedet::Facedet facedet_model(det_path);
    facenet::Facenet facenet(face_path);

    // 初始化属性模型
    attribute::Attribute gender_model(gender_path);
    attribute::Attribute age_model(age_path);
    attribute::Attribute emotion_model(emotion_path);
    attribute::Attribute mask_model(mask_path);

    facenet.loadSavedFeatures(FLAGS_faces, &facedet_model);

    cv::VideoCapture cap;
    if (vid.empty())
    {
        cap.open(2);
    }
    else
    {
        cap.open(vid, cv::CAP_FFMPEG);
    }

    // get height with from cap
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = 25;
    std::cout << "width: " << width << ", height: " << height << ", fps: " << fps << std::endl;

    // -> mp4
    // cv::VideoWriter writer;
    // int codec = cv::VideoWriter::fourcc('H','2','6','4');
    // writer.open("output.mp4", codec, fps, cv::Size(width, height), true);
    // if (!writer.isOpened()) {
    //     std::cerr << "Error: Could not open the output video file for writing." << std::endl;
    //     return -1;
    // }
    // -> streamer
    streamer::Streamer streamer;
    streamer::StreamerConfig streamer_config(width, height, width, height,
                                             fps, 500000, "main", "rtmp://localhost/live");
    streamer.init(streamer_config);
    cv::Mat frame;
    while(cap.read(frame)){
        // FPS start time
        auto start = std::chrono::high_resolution_clock::now();
        // detect face
        auto faces = facedet_model.run(frame);
        std::vector<DetWithAttr> dets;
        for (auto &face : faces){
            cv::Mat crop_face = frame(getRect(frame, face, facedet::kInputH, facedet::kInputW)); // 裁剪人脸
            auto name = facenet.run(crop_face);                                                  // 识别人脸

            auto gender_label = gender_model.run(crop_face, attribute::GENDER);    // 性别识别
            auto age_label = age_model.run(crop_face, attribute::AGE);             // 年龄识别
            auto emotion_label = emotion_model.run(crop_face, attribute::EMOTION); // 表情识别
            auto mask_label = mask_model.run(crop_face, attribute::MASK);          // 口罩识别

            // print info
            std::cout << "name: " << name << ", gender: " << gender_label << ", age: " << age_label
                      << ", emotion: " << emotion_label << ", mask: " << mask_label << std::endl;

            DetWithAttr face_attr{face, name, gender_label, age_label, emotion_label, mask_label};
            dets.push_back(face_attr);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f;

        draw(frame, elapsed, dets, facedet::kInputH, facedet::kInputW);
        streamer.stream_frame(frame.data);
        // writer.write(frame);
    }

    // writer.release();
    return 0;
}


