#include "postprocess.h"

#include <map>
#include <vector>
#include <algorithm>

bool cmp(const Detection &a, const Detection &b)
{
    return a.conf > b.conf;
}


float iou(float lbox[4], float rbox[4]){
    float interBox[] = {
        (std::max)(lbox[0], rbox[0]), //left
        (std::min)(lbox[2], rbox[2]), //right
        (std::max)(lbox[1], rbox[1]), //top
        (std::min)(lbox[3], rbox[3]) //bottom
    };
  if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
    return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / (
        ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + 
        (rbox[2] - rbox[0]) * (rbox[3] - rbox[1])) -
        interBoxS
    );
}


std::vector<Detection> nms(std::vector<Detection> &dets, float thresh){
    std::vector<Detection> out_dets;
    std::map<int32_t, std::vector<Detection>> m;
    for(auto &det: dets){
        if(m.count(det.class_id) == 0){
            m.emplace(det.class_id, std::vector<Detection>());
        }
        m[det.class_id].push_back(det);
    }
    for(auto it = m.begin(); it != m.end(); it++){
        auto &dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for(size_t i = 0; i < dets.size(); ++i){
            auto &item = dets[i];
            out_dets.push_back(item);
            for(size_t j = i + 1; j < dets.size(); ++j)
            {
                if(iou(item.bbox, dets[j].bbox) > thresh){
                    dets.erase(dets.begin() + j);
                    --j;
                }
            }
        }
    }
    return out_dets;
}

