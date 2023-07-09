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
            const context& ctxt,
            scene* scene,
            aten::hitrecord* first_hrec = nullptr);

        static void radiance_with_feature_line(
            int32_t idx,
            Path& paths,
            const context& ctxt,
            ray* rays,
            int32_t rrDepth,
            int32_t startDepth,
            int32_t maxDepth,
            camera* cam,
            scene* scene,
            const background* bg);

        static bool shade(
            int32_t idx,
            aten::Path& paths,
            const context& ctxt,
            ray* rays,
            const aten::hitrecord& rec,
            scene* scene,
            int32_t rrDepth,
            int32_t depth);

        static void shadeMiss(
            int32_t idx,
            scene* scene,
            int32_t depth,
            Path& paths,
            const ray* rays,
            const background* bg);

    protected:
        Path paths_;
        std::vector<aten::ray> rays_;

        int32_t m_maxDepth{ 1 };

        // Depth to compute russinan roulette.
        int32_t m_rrDepth{ 1 };

        int32_t m_startDepth{ 0 };

        std::vector<std::shared_ptr<texture>> m_noisetex;

        bool enable_feature_line_{ false };
    };
}
