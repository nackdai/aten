#pragma once

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "aten4idaten.h"

namespace idaten {
    namespace _detail {
        using v4 = float4;
        using v3 = float3;
    };

    struct PathThroughput {
        aten::vec3 throughput;
        real pdfb;
    };

    struct PathContrib {
        union {
            _detail::v4 v;
            struct {
                _detail::v3 contrib;
                float samples;
            };
        };
#ifndef __AT_CUDA__
        PathContrib() : contrib(0), samples(0.0f) {}

        PathContrib(const PathContrib& rhs)
        {
            v = rhs.v;
        }
        PathContrib(PathContrib&& rhs) noexcept
        {
            v = rhs.v;
        }
        PathContrib& operator=(const PathContrib& rhs)
        {
            v = rhs.v;
        }
        PathContrib& operator=(PathContrib&& rhs) noexcept
        {
            v = rhs.v;
        }
        ~PathContrib() = default;
#endif
    };

    struct PathAttribute {
        union {
            _detail::v4 v;
            struct {
                bool isHit;
                bool isTerminate;
                bool isSingular;
                bool isKill;

                aten::MaterialType mtrlType;
            };
        };

#ifndef __AT_CUDA__
        PathAttribute() : isHit(false), isTerminate(false), isSingular(false), isKill(false), mtrlType(aten::MaterialType::Lambert) {}
        PathAttribute(const PathAttribute& rhs)
        {
            v = rhs.v;
        }
        PathAttribute(PathAttribute&& rhs) noexcept
        {
            v = rhs.v;
        }
        PathAttribute& operator=(const PathAttribute& rhs)
        {
            v = rhs.v;
        }
        PathAttribute& operator=(PathAttribute&& rhs) noexcept
        {
            v = rhs.v;
        }
        ~PathAttribute() = default;
#endif
    };

    struct Path {
        PathThroughput* throughput;
        PathContrib* contrib;
        PathAttribute* attrib;
        aten::sampler* sampler;
    };

    struct PathHost {
        Path paths;

        idaten::TypedCudaMemory<PathThroughput> throughput;
        idaten::TypedCudaMemory<PathContrib> contrib;
        idaten::TypedCudaMemory<PathAttribute> attrib;
        idaten::TypedCudaMemory<aten::sampler> sampler;

        bool init(int32_t width, int32_t height)
        {
            if (throughput.empty()) {
                throughput.init(width * height);
                contrib.init(width * height);
                attrib.init(width * height);
                sampler.init(width * height);

                paths.throughput = throughput.ptr();
                paths.contrib = contrib.ptr();
                paths.attrib = attrib.ptr();
                paths.sampler = sampler.ptr();

                return true;
            }
            return false;
        }
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
