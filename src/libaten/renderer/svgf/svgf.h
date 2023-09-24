#pragma once

#include "camera/camera.h"
#include "misc/span.h"
#include "renderer/pathtracing/pt_params.h"
#include "renderer/renderer.h"
#include "renderer/svgf/svgf_types.h"
#include "scene/scene.h"

namespace aten
{
    class SVGFRenderer : public Renderer {
    public:
        SVGFRenderer() = default;
        ~SVGFRenderer() = default;

        SVGFRenderer(const SVGFRenderer&) = delete;
        SVGFRenderer(SVGFRenderer&&) = delete;
        SVGFRenderer& operator=(const SVGFRenderer&) = delete;
        SVGFRenderer& operator=(SVGFRenderer&&) = delete;

        virtual void onRender(
            const context& ctxt,
            Destination& dst,
            scene* scene,
            camera* camera) override;

    private:
        void Initialize(
            const Destination& dst,
            const camera& camera);

        void ExecRendering(
            int32_t idx,
            int32_t ix, int32_t iy,
            int32_t width, int32_t height,
            const context& ctxt,
            scene* scene,
            const aten::CameraParameter& camera);

        static void Shade(
            int32_t idx,
            aten::Path& paths,
            const context& ctxt,
            ray* rays,
            aten::ShadowRay* shadow_rays,
            const aten::Intersection& isect,
            scene* scene,
            int32_t rrDepth,
            int32_t bounce,
            const AT_NAME::SVGFMtxPack mtxs,
            aten::span<aten::vec4>& aov_normal_depth,
            aten::span<aten::vec4>& aov_albedo_meshid);

    private:
        PathHost path_host_;
        std::vector<aten::ray> rays_;
        std::vector<aten::ShadowRay> shadow_rays_;

        int32_t max_depth_{ 0 };
        int32_t russian_roulette_depth_{ 0 };

        AT_NAME::SVGFParams<std::vector<aten::vec4>> params_;
    };
}
