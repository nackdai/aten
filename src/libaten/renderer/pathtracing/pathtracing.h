#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "light/pointlight.h"
#include "renderer/pathtracing/pt_params.h"
#include "renderer/npr/feature_line.h"

namespace aten
{
    class PathTracing : public Renderer {
    public:
        PathTracing() = default;
        ~PathTracing() = default;

        virtual void OnRender(
            const context& ctxt,
            Destination& dst,
            scene* scene,
            Camera* camera) override;

        void registerBlueNoiseTex(const std::shared_ptr<texture>& tex)
        {
            m_noisetex.push_back(tex);
        }

        virtual void enableFeatureLine(bool e) override {
            enable_feature_line_ = e;
        }

        static void shade(
            int32_t idx,
            aten::Path& paths,
            const context& ctxt,
            ray* rays,
            aten::ShadowRay* shadow_rays,
            const aten::Intersection& isect,
            scene* scene,
            int32_t rrDepth,
            int32_t bounce);

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

        static void shadeMiss(
            const aten::context& ctxt,
            int32_t idx,
            scene* scene,
            int32_t depth,
            Path& paths,
            const ray* rays,
            aten::BackgroundResource& bg);

    protected:
        PathHost path_host_;
        std::vector<aten::ray> rays_;

        std::vector<aten::ShadowRay> shadow_rays_;

        static constexpr size_t SampleRayNum = 8;
        std::vector<aten::npr::FeatureLine::SampleRayInfo<SampleRayNum>> feature_line_sample_ray_infos_;

        int32_t m_maxDepth{ 1 };

        // Depth to compute russinan roulette.
        int32_t m_rrDepth{ 1 };

        std::vector<std::shared_ptr<texture>> m_noisetex;

        bool enable_feature_line_{ false };
    };
}
