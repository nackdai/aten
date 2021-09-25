#pragma once

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "aten4idaten.h"

namespace idaten {
    struct PathThroughput {
        aten::vec3 throughput;
        real pdfb;
    };

    struct PathContrib {
        union {
            float4 v;
            struct {
                float3 contrib;
                float samples;
            };
        };
    };

    struct PathAttribute {
        union {
            float4 v;
            struct {
                bool isHit;
                bool isTerminate;
                bool isSingular;
                bool isKill;

                aten::MaterialType mtrlType;
            };
        };
    };

    struct Path {
        PathThroughput* throughput;
        PathContrib* contrib;
        PathAttribute* attrib;
        aten::sampler* sampler;
    };

    struct ShadowRay {
        aten::vec3 rayorg;
        real distToLight;

        aten::vec3 raydir;
        struct {
            uint32_t isActive : 1;
        };

        aten::vec3 lightcontrib;
        uint32_t targetLightId;
    };

    struct ReSTIRIntermedidate {
        aten::vec3 light_sample_nml;
        struct {
            uint32_t is_voxel : 1;
            uint32_t is_mtrl_valid : 1;
            uint32_t mtrl_idx : 16;
        };

        aten::vec3 light_color;
        float nml_x;

        aten::vec3 wi;
        float nml_y;

        aten::vec3 throughput;
        float nml_z;

        __host__ __device__ void setNml(const aten::vec3& nml)
        {
            nml_x = nml.x;
            nml_y = nml.y;
            nml_z = nml.z;
        }

        __host__ __device__ void getNml(aten::vec3& nml) const
        {
            nml.x = nml_x;
            nml.y = nml_y;
            nml.z = nml_z;
        }

        __host__ __device__ void clear()
        {
            light_sample_nml.x = light_sample_nml.y = light_sample_nml.z = 0.0f;

            is_voxel = false;
            is_mtrl_valid = false;
            mtrl_idx = 0;

            light_color.x = light_color.y = light_color.z = 0.0f;
            wi.x = wi.y = wi.z = 0.0f;
            throughput.x = throughput.y = throughput.z = 0.0f;

            nml_x = nml_y = nml_z = 0.0f;
        }
    };

    struct Reservoir {
        float w;
        float m;
        float light_pdf;
        int light_idx;
    };
}
