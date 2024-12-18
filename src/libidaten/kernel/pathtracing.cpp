#include "kernel/pathtracing.h"
#include "kernel/device_scene_context.cuh"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "kernel/StreamCompaction.h"
#include "kernel/pt_standard_impl.h"

#include "aten4idaten.h"
#include "renderer/pathtracing/pt_params.h"

//#pragma optimize( "", off)

namespace idaten
{
    void PathTracing::UpdateSceneData(
        GLuint gltex,
        int32_t width, int32_t height,
        const aten::CameraParameter& camera,
        const aten::context& scene_ctxt,
        const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
        uint32_t advance_prim_num,
        uint32_t advance_vtx_num,
        const aten::BackgroundResource& bg_resource,
        std::function<const aten::Grid* (const aten::context&)> proxy_get_grid_from_host_scene_context/*= nullptr*/)
    {
        Renderer::UpdateSceneData(
            gltex,
            width, height,
            camera, scene_ctxt, nodes,
            advance_prim_num, advance_vtx_num,
            bg_resource,
            proxy_get_grid_from_host_scene_context);

        initSamplerParameter(width, height);

        aov_.traverse([width, height](auto& buffer) {
            buffer.resize(width * height);
            });
    }

    void PathTracing::updateMaterial(const std::vector<aten::MaterialParameter>& mtrls)
    {
        AT_ASSERT(mtrls.size() <= ctxt_host_->mtrlparam.num());

        if (mtrls.size() <= ctxt_host_->mtrlparam.num()) {
            ctxt_host_->mtrlparam.writeFromHostToDeviceByNum(&mtrls[0], (uint32_t)mtrls.size());
            reset();
        }
    }

    void PathTracing::updateLight(const aten::context& scene_ctxt)
    {
        const auto lights = scene_ctxt.GetLightParameters();

        AT_ASSERT(lights.size() <= ctxt_host_->lightparam.num());

        if (lights.size() <= ctxt_host_->lightparam.num()) {
            ctxt_host_->lightparam.writeFromHostToDeviceByNum(&lights[0], (uint32_t)lights.size());
            reset();
        }
    }

    static bool doneSetStackSize = false;

    void PathTracing::render(
        int32_t width, int32_t height,
        int32_t maxSamples,
        int32_t maxBounce)
    {
#ifdef __AT_DEBUG__
        if (!doneSetStackSize) {
            size_t val = 0;
            cudaThreadGetLimit(&val, cudaLimitStackSize);
            cudaThreadSetLimit(cudaLimitStackSize, val * 4);
            doneSetStackSize = true;
        }
#endif

        int32_t bounce = 0;

        m_isects.resize(width * height);
        m_rays.resize(width * height);

        m_shadowRays.resize(width * height);

        InitPath(width, height);

        CudaGLResourceMapper<decltype(m_glimg)> rscmap(m_glimg);
        auto outputSurf = m_glimg.bind();

        m_hitbools.resize(width * height);
        m_hitidx.resize(width * height);

        m_compaction.init(
            width * height,
            1024);

        clearPath();

        OnRender(
            width, height, maxSamples, maxBounce,
            outputSurf);

        m_frame++;
    }

    void PathTracing::OnRender(
        int32_t width, int32_t height,
        int32_t maxSamples,
        int32_t maxBounce,
        cudaSurfaceObject_t outputSurf)
    {
        static const int32_t rrBounce = 3;

        // Set bounce count to 1 forcibly, AOV rendering mode.
        maxBounce = (m_mode == Mode::AOV ? 1 : maxBounce);

        auto time = AT_NAME::timer::getSystemTime();

        for (int32_t i = 0; i < maxSamples; i++) {
            int32_t seed = time.milliSeconds;
            //int32_t seed = 0;

            generatePath(
                width, height,
                m_mode == Mode::AOV,
                i, maxBounce,
                seed);

            int32_t bounce = 0;

            while (bounce < maxBounce) {
                onHitTest(
                    width, height,
                    bounce);

                missShade(width, height, bounce);

                int32_t hitcount = 0;
                m_compaction.compact(
                    m_hitidx,
                    m_hitbools);

                //AT_PRINTF("%d\n", hitcount);

                onShade(
                    outputSurf,
                    width, height,
                    i,
                    bounce, rrBounce, maxBounce);

                bounce++;
            }
        }

        if (m_mode == Mode::PT) {
            onGather(outputSurf, width, height, maxSamples);
        }
        else if (m_mode == Mode::AOV) {
            DisplayAOV(outputSurf, width, height);
        }
        else {
            AT_ASSERT(false);
        }
    }

    void PathTracing::SetStream(cudaStream_t stream)
    {
        m_stream = stream;
        m_compaction.SetStream(stream);
    }
}
