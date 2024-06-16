#pragma once
#include <string>

struct alignas(float) Detection {
    float bbox[4]; // xmin ymin xmax ymax
    float conf;
    float class_id;
};

struct DetWithAttr{
    Detection det;
    std::string name;
    std::string gender;
    std::string age;
    std::string emotion;
    std::string mask;
};