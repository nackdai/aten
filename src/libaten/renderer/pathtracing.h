#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "light/pointlight.h"

namespace aten
{
    class PathTracing : public Renderer {
    public:
        PathTracing() {}
        ~PathTracing() {}

        virtual void onRender(
            const context& ctxt,
            Destination& dst,
            scene* scene,
            camera* camera) override;

        void setVirtualLight(const vec3& pos, const vec3& dir, const vec3& le)
        {
            m_virtualLight = new PointLight(pos, le);
            m_lightDir = dir;
        }

        void setVirtualLight(PointLight* light, const vec3& dir)
        {
            m_virtualLight = light;
            m_lightDir = dir;
        }

        void registerBlueNoiseTex(const std::shared_ptr<texture> tex)
        {
            m_noisetex.push_back(tex);
        }

    protected:
        struct Path {
            vec3 contrib{ real(0) };
            vec3 throughput{ real(1) };
            real pdfb{ real(1) };
            real accumulatedAlpha{ real(1) };

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

        PointLight* m_virtualLight{ nullptr };
        vec3 m_lightDir;

        std::vector<std::shared_ptr<texture>> m_noisetex;
    };
}
