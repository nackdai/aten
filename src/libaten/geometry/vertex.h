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

        // z == 1, compute plane normal in real-time.
        // z == -1, there is no texture coordinate.
        vec3 uv;
    };

    struct CompressedVertex {
        // NOTE
        // pos.w == uv.x, nml.w == uv.y.
        vec4 pos;
        vec4 nml;
    };
}
