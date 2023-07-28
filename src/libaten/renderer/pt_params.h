#pragma once

#include "material/material.h"
#include "math/vec3.h"
#include "math/vec4.h"
#include "sampler/cmj.h"

namespace aten {
    struct PathThroughput {
        aten::vec3 throughput;
        real pdfb;
    };

    union PathContrib {
        aten::vec4 v;
        struct {
            aten::vec3 contrib;
            float samples;
        };
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
    };

    union PathAttribute {
        aten::vec4 v;
        struct {
            bool isHit;
            bool isTerminate;
            bool isSingular;
            bool isKill;

            aten::MaterialType mtrlType;
        };

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
    };

    struct Path {
        PathThroughput* throughput;
        PathContrib* contrib;
        PathAttribute* attrib;
        aten::sampler* sampler;

        void init(int32_t width, int32_t height)
        {
            if (throughput_.empty()) {
                throughput_.resize(width * height);
                contrib_.resize(width * height);
                attrib_.resize(width * height);
                sampler_.resize(width * height);

                throughput = throughput_.data();
                contrib = contrib_.data();
                attrib = attrib_.data();
                sampler = sampler_.data();
            }
        }

    private:
        std::vector<PathThroughput> throughput_;
        std::vector<PathContrib> contrib_;
        std::vector<PathAttribute> attrib_;
        std::vector<aten::sampler> sampler_;
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
