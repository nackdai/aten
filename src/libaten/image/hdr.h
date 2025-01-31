#pragma once

#include <string>
#include <vector>
#include "math/vec3.h"
#include "math/vec4.h"

namespace aten
{
    class HDRExporter {
    public:
        static bool save(
            const std::string& filename,
            const vec4* image,
            const int32_t width, const int32_t height);
    };
}
