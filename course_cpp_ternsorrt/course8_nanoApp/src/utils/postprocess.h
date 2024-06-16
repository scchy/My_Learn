#pragma once

#include <vector>

#include "types.h"

std::vector<Detection> nms(std::vector<Detection>& dets, float thresh);

