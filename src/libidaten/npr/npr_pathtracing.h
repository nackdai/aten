#pragma once

#include "kernel/pathtracing.h"

namespace idaten
{
    class NPRPathTracing : public PathTracing {
    public:
        NPRPathTracing() = default;
        virtual ~NPRPathTracing() = default;

        static constexpr size_t SampleRayNum = 8;

    protected:
        void onShade(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            int32_t sample,
            int32_t bounce, int32_t rrBounce, int32_t max_depth) override;

        virtual void missShade(
            int32_t width, int32_t height,
            int32_t bounce) override;

    protected:
        idaten::TypedCudaMemory<AT_NAME::npr::FeatureLine::SampleRayInfo<SampleRayNum>> sample_ray_infos_;
    };
}
