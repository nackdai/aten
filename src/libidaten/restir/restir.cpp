#include "restir/restir.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "kernel/StreamCompaction.h"
#include "kernel/pt_standard_impl.h"

#include "aten4idaten.h"

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
        const EnvmapResource& envmapRsc)
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
            texs, envmapRsc);

        initSamplerParameter(width, height);

        aov_.traverse([&width, &height](auto& buffer) { buffer.init(width * height); });
    }

    void ReSTIRPathTracing::setGBuffer(
        GLuint gltexGbuffer,
        GLuint gltexMotionDepthbuffer)
    {
        m_gbuffer.init(gltexGbuffer, idaten::CudaGLRscRegisterType::ReadOnly);
        m_motionDepthBuffer.init(gltexMotionDepthbuffer, idaten::CudaGLRscRegisterType::ReadOnly);
    }

    static bool doneSetStackSize = false;

    void ReSTIRPathTracing::render(
        const TileDomain& tileDomain,
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

        int32_t width = tileDomain.w;
        int32_t height = tileDomain.h;

        m_isects.init(width * height);
        m_rays.init(width * height);

        m_shadowRays.init(width * height);

        initPath(width, height);

        CudaGLResourceMapper<decltype(m_glimg)> rscmap(m_glimg);
        auto outputSurf = m_glimg.bind();

        auto vtxTexPos = m_vtxparamsPos.bind();
        auto vtxTexNml = m_vtxparamsNml.bind();

        // TODO
        // Textureメモリのバインドによる取得されるcudaTextureObject_tは変化しないので,値を一度保持しておけばいい.
        // 現時点では最初に設定されたものが変化しない前提でいるが、入れ替えなどの変更があった場合はこの限りではないので、何かしらの対応が必要.

        if (!m_isListedTextureObject)
        {
            {
                std::vector<cudaTextureObject_t> tmp;
                for (int32_t i = 0; i < m_nodeparam.size(); i++) {
                    auto nodeTex = m_nodeparam[i].bind();
                    tmp.push_back(nodeTex);
                }
                m_nodetex.writeFromHostToDeviceByNum(&tmp[0], tmp.size());
            }

            if (!m_texRsc.empty())
            {
                std::vector<cudaTextureObject_t> tmp;
                for (int32_t i = 0; i < m_texRsc.size(); i++) {
                    auto cudaTex = m_texRsc[i].bind();
                    tmp.push_back(cudaTex);
                }
                m_tex.writeFromHostToDeviceByNum(&tmp[0], tmp.size());
            }

            m_isListedTextureObject = true;
        }
        else {
            for (int32_t i = 0; i < m_nodeparam.size(); i++) {
                auto nodeTex = m_nodeparam[i].bind();
            }
            for (int32_t i = 0; i < m_texRsc.size(); i++) {
                auto cudaTex = m_texRsc[i].bind();
            }
        }

        m_hitbools.init(width * height);
        m_hitidx.init(width * height);

        m_compaction.init(
            width * height,
            1024);

        clearPath();

        onRender(
            tileDomain,
            width, height, maxSamples, maxBounce,
            outputSurf,
            vtxTexPos,
            vtxTexNml);

        {
            m_mtxPrevW2V = m_mtxW2V;

            pick(
                m_pickedInfo.ix, m_pickedInfo.iy,
                width, height,
                vtxTexPos);

            //checkCudaErrors(cudaDeviceSynchronize());

            m_frame++;
        }
    }

    void ReSTIRPathTracing::onRender(
        const TileDomain& tileDomain,
        int32_t width, int32_t height,
        int32_t maxSamples,
        int32_t maxBounce,
        cudaSurfaceObject_t outputSurf,
        cudaTextureObject_t vtxTexPos,
        cudaTextureObject_t vtxTexNml)
    {
        m_tileDomain = tileDomain;

        static const int32_t rrBounce = 3;

        // Set bounce count to 1 forcibly, aov render mode.
        maxBounce = (m_mode == Mode::AOVar ? 1 : maxBounce);

        bool is_restir = m_mode == Mode::ReSTIR;

        auto time = AT_NAME::timer::getSystemTime();

        for (int32_t i = 0; i < maxSamples; i++) {
            int32_t seed = time.milliSeconds;
            //int32_t seed = 0;

            generatePath(
                m_mode == Mode::AOVar,
                i, maxBounce,
                seed);

            initReSTIR(width, height);

            int32_t bounce = 0;

            while (bounce < maxBounce) {
                onHitTest(
                    width, height,
                    bounce,
                    vtxTexPos);

                missShade(width, height, bounce);

                int32_t hitcount = 0;
                m_compaction.compact(
                    m_hitidx,
                    m_hitbools);

                //AT_PRINTF("%d\n", hitcount);

                if (is_restir && bounce == 0) {
                    onShadeReSTIR(
                        outputSurf,
                        width, height,
                        i,
                        bounce, rrBounce,
                        vtxTexPos, vtxTexNml);
                }
                else {
                    onShade(
                        outputSurf,
                        width, height,
                        i,
                        bounce, rrBounce,
                        vtxTexPos, vtxTexNml);
                }

                bounce++;
            }
        }

        if (m_mode == Mode::ReSTIR || m_mode == Mode::PT) {
            onGather(outputSurf, width, height, maxSamples);
        }
        else if (m_mode == Mode::AOVar) {
            onDisplayAOV(outputSurf, width, height, vtxTexPos);
        }
        else {
            AT_ASSERT(false);
        }
    }

    void ReSTIRPathTracing::setStream(cudaStream_t stream)
    {
        m_stream = stream;
        m_compaction.setStream(stream);
    }
}
