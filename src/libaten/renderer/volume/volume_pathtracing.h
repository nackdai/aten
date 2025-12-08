#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "light/pointlight.h"
#include "renderer/pathtracing/pt_params.h"

namespace aten
{
    class VolumePathTracing : public Renderer {
    public:
        VolumePathTracing() = default;
        ~VolumePathTracing() = default;

        virtual void OnRender(
            const context& ctxt,
            Destination& dst,
            scene* scene,
            Camera* camera) override;

        void EnableRenderGrid(bool enable)
        {
            is_render_grid_ = enable;
        }

    protected:
        static void Shade(
            int32_t idx,
            aten::Path& paths,
            const context& ctxt,
            ray* rays,
            const aten::Intersection& isect,
            scene* scene,
            int32_t rrDepth,
            int32_t bounce);

        static void Nee(
            int32_t idx,
            aten::Path& paths,
            const context& ctxt,
            ray* rays,
            aten::ShadowRay* shadow_rays,
            const aten::Intersection& isect,
            scene* scene,
            int32_t rrDepth);

        static void TraverseShadowRay(
            const int32_t idx,
            const int32_t max_depth,
            aten::Path& paths,
            const aten::context& ctxt,
            const aten::Intersection& isect,
            const aten::ShadowRay* shadow_rays,
            aten::scene* scene);

        void radiance(
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

        int32_t m_maxDepth{ 1 };

        // Depth to compute russinan roulette.
        int32_t m_rrDepth{ 1 };

        bool is_render_grid_{ false };
    };
}
