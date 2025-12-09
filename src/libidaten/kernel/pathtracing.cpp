#include "kernel/pathtracing.h"
#include "kernel/device_scene_context.cuh"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "kernel/StreamCompaction.h"

#include "aten4idaten.h"
#include "renderer/pathtracing/pt_params.h"

//#pragma optimize( "", off)

namespace idaten
{
    void PathTracingImplBase::SetGBuffer(GLuint gltexGbuffer)
    {
        g_buffer_.init(gltexGbuffer, idaten::CudaGLRscRegisterType::ReadOnly);
    }

    void PathTracing::UpdateSceneData(
        GLuint gltex,
        int32_t width, int32_t height,
        const aten::CameraParameter& camera,
        const aten::context& scene_ctxt,
        const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
        uint32_t advance_prim_num,
        uint32_t advance_vtx_num,
        std::function<const aten::Grid* (const aten::context&)> proxy_get_grid_from_host_scene_context/*= nullptr*/)
    {
        Renderer::UpdateSceneData(
            gltex,
            width, height,
            camera, scene_ctxt, nodes,
            advance_prim_num, advance_vtx_num,
            proxy_get_grid_from_host_scene_context);

        initSamplerParameter(width, height);

        aov_.traverse([width, height](auto& buffer) {
            buffer.resize(width * height);
            });
    }

    static bool doneSetStackSize = false;

    void PathTracingImplBase::render(
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

        m_glimg.unbind();
    }

    void PathTracing::OnRender(
        int32_t width, int32_t height,
        int32_t maxSamples,
        int32_t maxBounce,
        cudaSurfaceObject_t outputSurf)
    {
        static const int32_t rrBounce = 3;

        m_shadowRays.resize(width * height);

        // Set bounce count to 1 forcibly, AOV rendering mode.
        maxBounce = (m_mode == Mode::AOV ? 1 : maxBounce);

        auto seed = AT_NAME::timer::GetCurrMilliseconds();

        for (int32_t i = 0; i < maxSamples; i++) {
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
