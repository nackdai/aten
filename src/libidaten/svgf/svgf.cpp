#include "svgf/svgf.h"

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
    void SVGFPathTracing::UpdateSceneData(
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

        params_.InitBuffers(width, height);
    }

    void SVGFPathTracing::SetGBuffer(
        GLuint gltexGbuffer,
        GLuint gltexMotionDepthbuffer)
    {
        PathTracingImplBase::SetGBuffer(gltexGbuffer);
        params_.motion_depth_buffer.init(gltexMotionDepthbuffer, idaten::CudaGLRscRegisterType::ReadOnly);
    }

    static bool doneSetStackSize = false;

    void SVGFPathTracing::render(
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

        params_.mtxs.Reset(m_cam);

        clearPath();

        OnRender(
            width, height, maxSamples, maxBounce,
            outputSurf);

        onDenoise(
            width, height,
            outputSurf);

        if (m_mode == Mode::SVGF)
        {
            onAtrousFilter(outputSurf, width, height);
            onCopyFromTmpBufferToAov(width, height);
        }

        {
            pick(
                m_pickedInfo.ix, m_pickedInfo.iy,
                width, height);

            //checkCudaErrors(cudaDeviceSynchronize());

            // Toggle aov buffer pos.
            params_.UpdateCurrAovBufferPos();

            m_frame++;
        }
    }

    void SVGFPathTracing::OnRender(
        int32_t width, int32_t height,
        int32_t maxSamples,
        int32_t maxBounce,
        cudaSurfaceObject_t outputSurf)
    {
        static const int32_t rrBounce = 3;

        // Set bounce count to 1 forcibly, aov render mode.
        maxBounce = (m_mode == Mode::AOVar ? 1 : maxBounce);

        auto time = AT_NAME::timer::getSystemTime();

        for (int32_t i = 0; i < maxSamples; i++) {
            int32_t seed = time.milliSeconds;
            //int32_t seed = 0;

            generatePath(
                width, height,
                m_mode == Mode::AOVar,
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
        else if (m_mode == Mode::AOVar) {
            OnDisplayAOV(outputSurf, width, height);
        }
        else {
            onGather(outputSurf, width, height, maxSamples);
        }
    }

    void SVGFPathTracing::onDenoise(
        int32_t width, int32_t height,
        cudaSurfaceObject_t outputSurf)
    {
        if (m_mode == Mode::SVGF
            || m_mode == Mode::TF
            || m_mode == Mode::VAR)
        {
            if (IsFirstFrame()) {
                // Nothing is done...
            }
            else {
                onTemporalReprojection(
                    outputSurf,
                    width, height);
            }
        }

        if (m_mode == Mode::SVGF
            || m_mode == Mode::VAR)
        {
            onVarianceEstimation(outputSurf, width, height);
        }
    }

    void SVGFPathTracing::SetStream(cudaStream_t stream)
    {
        m_stream = stream;
        m_compaction.SetStream(stream);
    }
}
