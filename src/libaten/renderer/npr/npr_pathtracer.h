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
        bool RenderPerSample(
            int32_t x, int32_t y,
            int32_t width, int32_t height,
            int32_t sample,
            aten::scene* scene,
            const aten::context& ctxt,
            const aten::Camera* camera);

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

        static void HitShadowRayWithKeepingIfHitToLight(
            int32_t idx, int32_t bounce,
            const aten::context& ctxt,
            const aten::Path& paths,
            const aten::Intersection& isect,
            std::vector<aten::ShadowRay>& shadowRays);

        template <class SrcType>
        void ApplyBilateralFilter(
            int32_t ix, int32_t iy,
            int32_t width, int32_t height,
            const std::vector<aten::Intersection>& isects,
            const SrcType* src,
            std::function<void(float)> dst);

    protected:
        static constexpr size_t SampleRayNum = 8;
        std::vector<aten::npr::FeatureLine::SampleRayInfo<SampleRayNum>> feature_line_sample_ray_infos_;

        std::vector<aten::Intersection> isects_;
        std::vector<aten::vec4> contributes_;
    };
}
