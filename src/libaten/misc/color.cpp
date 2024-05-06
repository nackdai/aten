#include "misc/color.h"

namespace AT_NAME {
    const aten::vec3 color::RGB2Y = aten::vec3(float(0.29900), float(0.58700), float(0.11400));
    const aten::vec3 color::RGB2Cb = aten::vec3(float(-0.16874), float(-0.33126), float(0.50000));
    const aten::vec3 color::RGB2Cr = aten::vec3(float(0.50000), float(-0.41869), float(-0.08131));
    const aten::vec3 color::YCbCr2R = aten::vec3(float(1.00000), float(0.00000), float(1.40200));
    const aten::vec3 color::YCbCr2G = aten::vec3(float(1.00000), float(-0.34414), float(-0.71414));
    const aten::vec3 color::YCbCr2B = aten::vec3(float(1.00000), float(1.77200), float(0.00000));
}
