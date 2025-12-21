#pragma once

#include "renderer/pathtracing/pathtracing.h"
#include "renderer/npr/feature_line.h"

namespace aten
{
    class NprPathTracer : public PathTracing {
    public:
        NprPathTracer() = default;
        ~NprPathTracer() = default;

        virtual void OnRender(
            context& ctxt,
            Destination& dst,
            scene* scene,
            Camera* camera) override;

    protected:
        void radiance(
            int32_t idx,
            int32_t ix, int32_t iy,
            int32_t width, int32_t height,
            const context& ctxt,
            scene* scene,
            const aten::CameraParameter& camera,
            aten::hitrecord* first_hrec = nullptr);

        void radiance_with_feature_line(
            int32_t idx,
            int32_t ix, int32_t iy,
            int32_t width, int32_t height,
            const context& ctxt,
            scene* scene,
            const aten::CameraParameter& camera);

        static void AdvanceNPRPath(
            const int32_t idx,
            const aten::context& ctxt,
            const aten::Path& paths,
            std::vector<aten::ray>& rays,
            aten::Intersection& isect);

    protected:
        static constexpr size_t SampleRayNum = 8;
        std::vector<aten::npr::FeatureLine::SampleRayInfo<SampleRayNum>> feature_line_sample_ray_infos_;
    };
}
