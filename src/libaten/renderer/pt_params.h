#pragma once

#include "material/material.h"
#include "math/vec3.h"
#include "math/vec4.h"
#include "sampler/sampler.h"

#ifdef __AT_CUDA__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#endif

namespace AT_NAME {
    namespace _detail {
#ifdef __AT_CUDA__
        using v4 = float4;
        using v3 = float3;
#else
        using v4 = aten::vec4;
        using v3 = aten::vec3;
#endif
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

        bool init(int32_t width, int32_t height)
        {
            if (throughput.empty()) {
                throughput.resize(width * height);
                contrib.resize(width * height);
                attrib.resize(width * height);
                sampler.resize(width * height);

                paths.throughput = throughput.data();
                paths.contrib = contrib.data();
                paths.attrib = attrib.data();
                paths.sampler = sampler.data();

                return true;
            }
            return false;
        }

#ifdef __AT_CUDA__
        idaten::TypedCudaMemory<PathThroughput> throughput;
        idaten::TypedCudaMemory<PathContrib> contrib;
        idaten::TypedCudaMemory<PathAttribute> attrib;
        idaten::TypedCudaMemory<aten::sampler> sampler;
#else
        std::vector<PathThroughput> throughput;
        std::vector<PathContrib> contrib;
        std::vector<PathAttribute> attrib;
        std::vector<aten::sampler> sampler;
#endif
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
