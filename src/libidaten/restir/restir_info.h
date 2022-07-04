#pragma once

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "aten4idaten.h"

namespace idaten {
    // NOTE
    // size of "bool" is expected as 1 byte.
    static_assert(sizeof(bool) == 1, "");

    struct ReSTIRInfo {
        aten::vec3 nml;
        int16_t mtrl_idx{ -1 };
        bool is_voxel{ false };
        uint8_t padding[3];

        aten::vec3 wi;
        float u;

        aten::vec3 throughput;
        float v;

        aten::vec3 p;
        float pre_sampled_r;

        __host__ __device__ void clear()
        {
            nml.x = nml.y = nml.z = 0.0f;

            is_voxel = false;
            mtrl_idx = -1;

            wi.x = wi.y = wi.z = 0.0f;
            throughput.x = throughput.y = throughput.z = 0.0f;

            u = v = 0.0f;

            pre_sampled_r = 0.0f;
        }

        __host__ __device__ bool isMtrlValid() const
        {
            return mtrl_idx >= 0;
        }
    };
}
