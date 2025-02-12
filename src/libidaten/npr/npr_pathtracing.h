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
            std::array<AT_NAME::npr::FeatureLine::SampleRayDesc, SampleRayNum> descs;
            AT_NAME::npr::FeatureLine::Disc disc;
        };

        bool isEnableFatureLine() const
        {
            return is_enable_feature_line_;
        }

        void enableFatureLine(bool b)
        {
            is_enable_feature_line_ = b;
        }

        float getFeatureLineWidth() const
        {
            return feature_line_width_;
        }

        void setFeatureLineWidth(float w)
        {
            feature_line_width_ = std::max(w, 1.0f);
        }

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
        idaten::TypedCudaMemory<SampleRayInfo> sample_ray_infos_;
        float feature_line_width_{ float(1) };
        bool is_enable_feature_line_{ false };
    };
}
