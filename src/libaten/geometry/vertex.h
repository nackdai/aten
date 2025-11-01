#pragma once

#include <vector>
#include "types.h"
#include "math/vec4.h"
#include "visualizer/GeomDataBuffer.h"

namespace aten
{
    struct vertex {
        vec4 pos;
        vec3 nml;

        // z == 1, compute plane normal in float-time.
        // z == -1, there is no texture coordinate.
        // It's used while loading mesh data. It's not used on the fly.
        vec3 uv;
    };
}
