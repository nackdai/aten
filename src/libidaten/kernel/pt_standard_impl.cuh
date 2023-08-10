#pragma once

#include "kernel/idatendefs.cuh"
#include "kernel/device_scene_context.cuh"
#include "kernel/accelerator.cuh"

#include "light/light_impl.h"
#include "renderer/pt_params.h"

namespace kernel {
    inline __device__ float executeRussianProbability(
        int32_t bounce,
        int32_t rrBounce,
        idaten::PathAttribute& path_attrib,
        idaten::PathThroughput& path_throughput,
        aten::sampler& sampler)
    {
        float russianProb = 1.0f;

        if (bounce > rrBounce) {
            auto t = normalize(path_throughput.throughput);
            auto p = aten::cmpMax(t.r, aten::cmpMax(t.g, t.b));

            russianProb = sampler.nextSample();

            if (russianProb >= p) {
                //shPaths[threadIdx.x].contrib = aten::vec3(0);
                path_attrib.isTerminate = true;
            }
            else {
                russianProb = max(p, 0.01f);
            }
        }

        return russianProb;
    }
}
