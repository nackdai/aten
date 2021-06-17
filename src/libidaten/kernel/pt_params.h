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
        int mtrl_idx;

        aten::vec3 light_color;
        struct {
            uint32_t is_voxel : 1;
        };

        aten::vec3 wi;
        float padding_1;

        aten::vec3 throughput;
        float padding_2;
    };

    struct Reservoir {
        float w;
        float m;
        float light_pdf;
        int light_idx;
    };
}
