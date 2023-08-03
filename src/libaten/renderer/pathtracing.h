#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "light/pointlight.h"
#include "renderer/pt_params.h"

namespace aten
{
    class PathTracing : public Renderer {
    public:
        PathTracing() = default;
        ~PathTracing() = default;

        virtual void onRender(
            const context& ctxt,
            Destination& dst,
            scene* scene,
            camera* camera) override;

        void registerBlueNoiseTex(const std::shared_ptr<texture>& tex)
        {
            m_noisetex.push_back(tex);
        }

        virtual void enableFeatureLine(bool e) override {
            enable_feature_line_ = e;
        }

    protected:
        void radiance(
            int32_t idx,
            int32_t ix, int32_t iy,
            int32_t width, int32_t height,
            const context& ctxt,
            scene* scene,
            const aten::CameraParameter& camera,
            aten::hitrecord* first_hrec = nullptr);

        static void radiance_with_feature_line(
            int32_t idx,
            Path& paths,
            const context& ctxt,
            ray* rays,
            aten::ShadowRay* shadow_rays,
            int32_t rrDepth,
            int32_t startDepth,
            int32_t maxDepth,
            camera* cam,
            scene* scene,
            const background* bg);

        static void shade(
            int32_t idx,
            aten::Path& paths,
            const context& ctxt,
            ray* rays,
            aten::ShadowRay* shadow_rays,
            const aten::hitrecord& rec,
            scene* scene,
            int32_t rrDepth,
            int32_t bounce);

        static void HitShadowRay(
            scene* scene,
            int32_t idx,
            int32_t bounce,
            const aten::context& ctxt,
            aten::Path paths,
            const aten::ShadowRay* shadow_rays);

        static void shadeMiss(
            int32_t idx,
            scene* scene,
            int32_t depth,
            Path& paths,
            const ray* rays,
            const background* bg);

    protected:
        PathHost path_host_;
        std::vector<aten::ray> rays_;

        std::vector<aten::ShadowRay> shadow_rays_;

        int32_t m_maxDepth{ 1 };

        // Depth to compute russinan roulette.
        int32_t m_rrDepth{ 1 };

        int32_t m_startDepth{ 0 };

        std::vector<std::shared_ptr<texture>> m_noisetex;

        bool enable_feature_line_{ false };
    };
}
