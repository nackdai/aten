#pragma once

#include <vector>
#include "geometry/geomparam.h"
#include "light/light.h"
#include "material/material.h"
#include "geometry/vertex.h"
#include "scene/scene.h"
#include "scene/context.h"

namespace aten {
    class DataCollector {
    private:
        DataCollector() {}
        ~DataCollector() {}

    public:
        static void collect(
            const context& ctxt,
            const scene& scene,
            std::vector<aten::ObjectParameter>& shapeparams,
            std::vector<aten::TriangleParameter>& primparams,
            std::vector<aten::LightParameter>& lightparams,
            std::vector<aten::MaterialParameter>& mtrlparms,
            std::vector<aten::vertex>& vtxparams);

        static void collectTriangles(
            const context& ctxt,
            std::vector<std::vector<aten::TriangleParameter>>& triangles,
            std::vector<int32_t>& triIdOffsets);
    };
}
