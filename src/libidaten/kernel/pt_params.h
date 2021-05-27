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
}
