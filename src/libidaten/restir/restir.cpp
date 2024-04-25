#include "restir/restir.h"

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
    void ReSTIRPathTracing::update(
        GLuint gltex,
        int32_t width, int32_t height,
        const aten::CameraParameter& camera,
        const std::vector<aten::ObjectParameter>& shapes,
        const std::vector<aten::MaterialParameter>& mtrls,
        const std::vector<aten::LightParameter>& lights,
        const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
        const std::vector<aten::TriangleParameter>& prims,
        uint32_t advancePrimNum,
        const std::vector<aten::vertex>& vtxs,
        uint32_t advanceVtxNum,
        const std::vector<aten::mat4>& mtxs,
        const std::vector<TextureResource>& texs,
        const aten::BackgroundResource& bg_resource)
    {
        idaten::Renderer::update(
            gltex,
            width, height,
            camera,
            shapes,
            mtrls,
            lights,
            nodes,
            prims, advancePrimNum,
            vtxs, advanceVtxNum,
            mtxs,
            texs, bg_resource);

        initSamplerParameter(width, height);

        aov_.traverse([&width, &height](auto& buffer) { buffer.resize(width * height); });
    }

    void ReSTIRPathTracing::SetGBuffer(
        GLuint gltexGbuffer,
        GLuint gltexMotionDepthbuffer)
    {
        m_gbuffer.init(gltexGbuffer, idaten::CudaGLRscRegisterType::ReadOnly);
        m_motionDepthBuffer.init(gltexMotionDepthbuffer, idaten::CudaGLRscRegisterType::ReadOnly);
    }

    static bool doneSetStackSize = false;

    void ReSTIRPathTracing::render(
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

        mtxs_.Reset(m_cam);

        clearPath();

        OnRender(
            width, height, maxSamples, maxBounce,
            outputSurf);

        {
            pick(
                m_pickedInfo.ix, m_pickedInfo.iy,
                width, height);

            //checkCudaErrors(cudaDeviceSynchronize());

            m_frame++;
        }
    }

    void ReSTIRPathTracing::OnRender(
        int32_t width, int32_t height,
        int32_t maxSamples,
        int32_t maxBounce,
        cudaSurfaceObject_t outputSurf)
    {
        static const int32_t rrBounce = 3;

        // Set bounce count to 1 forcibly, aov render mode.
        maxBounce = (m_mode == Mode::AOVar ? 1 : maxBounce);

        bool is_restir = m_mode == Mode::ReSTIR;

        auto time = AT_NAME::timer::getSystemTime();

        for (int32_t i = 0; i < maxSamples; i++) {
            int32_t seed = time.milliSeconds;
            //int32_t seed = 0;

            generatePath(
                width, height,
                m_mode == Mode::AOVar,
                i, maxBounce,
                seed);

            InitReSTIR(width, height);

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

                if (is_restir && bounce == 0) {
                    OnShadeReSTIR(
                        outputSurf,
                        width, height,
                        i,
                        bounce, rrBounce);
                }
                else {
                    onShade(
                        outputSurf,
                        width, height,
                        i,
                        bounce, rrBounce);
                }

                bounce++;
            }
        }

        if (m_mode == Mode::ReSTIR || m_mode == Mode::PT) {
            onGather(outputSurf, width, height, maxSamples);
        }
        else if (m_mode == Mode::AOVar) {
            OnDisplayAOV(outputSurf, width, height);
        }
        else {
            AT_ASSERT(false);
        }
    }

    void ReSTIRPathTracing::SetStream(cudaStream_t stream)
    {
        m_stream = stream;
        m_compaction.SetStream(stream);
    }
}
