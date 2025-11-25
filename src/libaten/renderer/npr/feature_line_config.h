#pragma once

#include "defs.h"
#include "math/vec3.h"

namespace aten {
    struct FeatureLineConfig {
        bool enabled{ false };
        aten::vec3 line_color{ 0.0F };
        float line_width{ 1.0F };
        float albedo_threshold{ 0.1F };
        float normal_threshold{ 0.1F };
    };
}
