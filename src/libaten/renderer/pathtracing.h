#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "light/pointlight.h"

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
        struct Path {
            vec3 contrib{ real(0) };
            vec3 throughput{ real(1) };
            real pdfb{ real(1) };

            hitrecord rec;
            std::shared_ptr<material> prevMtrl;

            aten::ray ray;

            bool isTerminate{ false };
        };

        Path radiance(
            const context& ctxt,
            sampler* sampler,
            const ray& inRay,
            camera* cam,
            CameraSampleResult& camsample,
            scene* scene)
        {
            return radiance(ctxt, sampler, m_maxDepth, inRay, cam, camsample, scene);
        }

        Path radiance(
            const context& ctxt,
            sampler* sampler,
            uint32_t maxDepth,
            const ray& inRay,
            camera* cam,
            CameraSampleResult& camsample,
            scene* scene,
            aten::hitrecord* first_hrec = nullptr);

        Path radiance_with_feature_line(
            const context& ctxt,
            sampler* sampler,
            uint32_t maxDepth,
            const ray& inRay,
            camera* cam,
            CameraSampleResult& camsample,
            scene* scene);

        bool shade(
            const context& ctxt,
            sampler* sampler,
            scene* scene,
            camera* cam,
            CameraSampleResult& camsample,
            int depth,
            Path& path);

        void shadeMiss(
            scene* scene,
            int depth,
            Path& path);

    protected:
        uint32_t m_maxDepth{ 1 };

        // Depth to compute russinan roulette.
        uint32_t m_rrDepth{ 1 };

        uint32_t m_startDepth{ 0 };

        std::vector<std::shared_ptr<texture>> m_noisetex;

        bool enable_feature_line_{ false };
    };
}
