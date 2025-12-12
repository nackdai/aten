#pragma once

#include "ao/ao.h"

namespace idaten
{
    class NPRPathTracing : public AORenderer {
    public:
        NPRPathTracing() = default;
        ~NPRPathTracing() = default;

        static constexpr size_t SampleRayNum = 8;

        enum HatchingShadow {
            None,
            AOBase,
            ShadowRayBase,
        };

        void SetHatchinShadow(HatchingShadow shadow)
        {
            hatching_shadow_ = shadow;
        }
        HatchingShadow GetHatchingShadow() const
        {
            return hatching_shadow_;
        }

    protected:
        void OnRender(
            int32_t width, int32_t height,
            int32_t maxSamples,
            int32_t maxBounce,
            cudaSurfaceObject_t outputSurf) override
        {
            PathTracing::OnRender(
                width, height,
                maxSamples,
                maxBounce,
                outputSurf);
        }

        void PreShade(
            int32_t width, int32_t height,
            int32_t bounce,
            cudaSurfaceObject_t outputSurf) override;

        void onShade(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            int32_t sample,
            int32_t bounce, int32_t rrBounce, int32_t max_depth) override;

        void onShadeByShadowRay(
            int32_t width, int32_t height,
            int32_t bounce) override;

        void missShade(
            int32_t width, int32_t height,
            int32_t bounce) override;

    protected:
        idaten::TypedCudaMemory<AT_NAME::npr::FeatureLine::SampleRayInfo<SampleRayNum>> sample_ray_infos_;

        idaten::TypedCudaMemory<float> ao_result_buffer_;
        HatchingShadow hatching_shadow_{ HatchingShadow::AOBase };
    };
}
