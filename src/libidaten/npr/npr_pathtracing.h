#pragma once

#include "kernel/pathtracing.h"

namespace idaten
{
    class NPRPathTracing : public PathTracing {
    public:
        NPRPathTracing() = default;
        virtual ~NPRPathTracing() = default;

        static constexpr size_t SampleRayNum = 8;

        struct SampleRayInfo {
            AT_NAME::FeatureLine::SampleRayDesc descs[SampleRayNum];
            AT_NAME::FeatureLine::Disc disc;
        };

    protected:
        virtual void onShade(
            cudaSurfaceObject_t outputSurf,
            int width, int height,
            int sample,
            int bounce, int rrBounce,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml) override;

        virtual void missShade(
            int width, int height,
            int bounce,
            int offsetX = -1,
            int offsetY = -1) override;

    protected:
        idaten::TypedCudaMemory<SampleRayInfo> sample_ray_infos_;
        real feature_line_width_{ real(1) };
    };
}
