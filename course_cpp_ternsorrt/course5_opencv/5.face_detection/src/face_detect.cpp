#include "opencv2/opencv.hpp"
#include <iostream>

// 初始化模型
const std::string tensorflowConfigFile = "./weights/opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = "./weights/opencv_face_detector_uint8.pb";

cv::dnn::Net net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);

// draw detectRect
void detectDrawRect(cv::Mat &frame){
    int fHeight = frame.rows;
    int fWidth = frame.cols;

    // preprocess: resize + swapRB + mean + scale
    cv::Mat inputBlob = cv::dnn::blobFromImage(
        frame, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0),
        false, false
    );
    // inference
    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");
    // get result
    cv::Mat detectionMat(
        detection.size[2], detection.size[3],
        CV_32F, detection.ptr<float>()
    );

    // for multi result
    for(int i = 0; i < detectionMat.rows; i++){
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > 0.2)
        {
            // point left top right bottom
            int l = static_cast<int>(detectionMat.at<float>(i, 3) * fWidth);
            int t = static_cast<int>(detectionMat.at<float>(i, 4) * fHeight);
            int r = static_cast<int>(detectionMat.at<float>(i, 5) * fWidth);
            int b = static_cast<int>(detectionMat.at<float>(i, 6) * fHeight);
            // draw rect
            cv::rectangle(
                frame, cv::Point(l, t), cv::Point(r, b),
                cv::Scalar(0, 255, 0), 2
            );
        }
    }

}

// test
void imageTest()
{
    // read pic
    cv::Mat img = cv::imread("./media/test_face.jpg");
    // cv::imshow("image org", img);
    // inference
    detectDrawRect(img);
    // imshow
    cv::imshow("image test", img);
    // save
    cv::imwrite("./output/test_face_result.jpg", img);
    cv::waitKey(0);
}

// video test
void videoTest()
{
    // camera
    // cv::VideoCapture cap(0);
    // int w = 640;
    // int h = 480;
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, w);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, h);

    // file
    cv::VideoCapture cap("./media/video.mp4");
    // video w h
    int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // writer sudo apt-get install libx264-dev
    // 写入MP4文件，参数分别是：文件名，编码格式，帧率，帧大小
    cv::VideoWriter writer(
        "./output/record.mp4", 
        cv::VideoWriter::fourcc('H', '2', '6', '4'),
        // cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        25,
        cv::Size(w, h)
    );
    if(!cap.isOpened()){
        std::cout << "Cannot open the video cam" << std::endl;
        // 退出
        exit(1);
    }
    cv::Mat frame;
    while(true)
    {
        if(!cap.read(frame))
        {
            std::cout << "Cannot read a frame from video stream" << std::endl;
            break;
        }
        cv::flip(frame, frame, 1);
        detectDrawRect(frame);
        writer.write(frame);
        cv::imshow("MyVideo", frame);
        if (cv::waitKey(1) == 27)
        {
            std::cout << "esc key is pressed by user" << std::endl;
            break;
        }
    }
}

int main(int argc, char **argv){
    // imageTest();
    videoTest();
    return 0;
}


