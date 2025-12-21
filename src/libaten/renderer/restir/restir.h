#pragma once

#include <optional>

#include "camera/camera.h"
#include "misc/span.h"
#include "renderer/aov.h"
#include "renderer/pathtracing/pt_params.h"
#include "renderer/renderer.h"
#include "renderer/restir/restir_types.h"
#include "scene/scene.h"
#include "visualizer/fbo.h"

namespace aten
{
    class ReSTIRRenderer : public Renderer {
    public:
        ReSTIRRenderer() = default;
        ~ReSTIRRenderer() = default;

        ReSTIRRenderer(const ReSTIRRenderer&) = delete;
        ReSTIRRenderer(ReSTIRRenderer&&) = delete;
        ReSTIRRenderer& operator=(const ReSTIRRenderer&) = delete;
        ReSTIRRenderer& operator=(ReSTIRRenderer&&) = delete;

        void OnRender(
            context& ctxt,
            Destination& dst,
            scene* scene,
            Camera* camera) override;

        void SetMotionDepthBuffer(aten::FBO& fbo, int32_t idx);

    private:
        void Initialize(
            const Destination& dst,
            const Camera& camera);

        void Render(
            int32_t idx,
            int32_t x, int32_t y,
            int32_t sample_cnt,
            int32_t bounce,
            int32_t width, int32_t height,
            const context& ctxt,
            scene* scene,
            const aten::CameraParameter& camera);

        static void Shade(
            int32_t idx,
            int32_t width, int32_t height,
            aten::Path& paths,
            const context& ctxt,
            ray* rays,
            const aten::Intersection& isect,
            aten::span<AT_NAME::Reservoir>& reservoirs,
            aten::span<AT_NAME::ReSTIRInfo>& restir_infos,
            int32_t rrDepth,
            int32_t bounce,
            const aten::mat4& mtx_W2C,
            aten::span<aten::vec4>& aov_normal_depth,
            aten::span<aten::vec4>& aov_albedo_meshid);

        static void EvaluateVisibility(
            int32_t idx,
            int32_t bounce,
            int32_t width, int32_t height,
            aten::Path& paths,
            const context& ctxt,
            aten::scene* scene,
            std::vector<AT_NAME::Reservoir>& reservoirs,
            std::vector<AT_NAME::ReSTIRInfo>& restir_infos,
            std::vector<aten::ShadowRay>& shadowRays);

        static void ApplySpatialReuse(
            int32_t idx,
            int32_t width, int32_t height,
            const aten::context& ctxt,
            aten::sampler& sampler,
            const std::vector<AT_NAME::Reservoir>& curr_reservoirs,
            std::vector<AT_NAME::Reservoir>& dst_reservoirs,
            const std::vector<AT_NAME::ReSTIRInfo>& infos,
            const std::vector<aten::vec4>& aovTexclrMeshid);

        static void ApplyTemporalReuse(
            int32_t idx,
            int32_t width, int32_t height,
            const aten::context& ctxt,
            aten::sampler& sampler,
            std::vector<AT_NAME::Reservoir>& reservoir,
            const std::vector<AT_NAME::Reservoir>& prev_reservoirs,
            const std::vector<AT_NAME::ReSTIRInfo>& curr_infos,
            const std::vector<AT_NAME::ReSTIRInfo>& prev_infos,
            const std::vector<aten::vec4>& aovTexclrMeshid,
            const std::vector<aten::vec4>& motion_depth_buffer);

        static void ComputePixelColor(
            int32_t idx,
            AT_NAME::Path& paths,
            const aten::context& ctxt,
            const std::vector<AT_NAME::Reservoir>& reservoirs,
            const std::vector<AT_NAME::ReSTIRInfo>& restir_infos,
            const std::vector<aten::vec4>& aov_albedo_meshid);

    private:
        PathHost path_host_;
        std::vector<aten::ray> rays_;
        std::vector<aten::ShadowRay> shadow_rays_;

        int32_t max_depth_{ 0 };
        int32_t russian_roulette_depth_{ 0 };

        AT_NAME::ReuseParams<std::vector<AT_NAME::Reservoir>> reservoirs_;
        AT_NAME::ReuseParams<std::vector<AT_NAME::ReSTIRInfo>> restir_infos_;
        AT_NAME::MatricesForRendering mtxs_;

        // AOV buffer
        using AOVHostBuffer = AT_NAME::AOVHostBuffer<std::vector<aten::vec4>, AT_NAME::AOVBufferType::NumBasicAovBuffer>;
        AOVHostBuffer aov_;

        std::vector<aten::vec4> motion_depth_buffer_;
    };
}
