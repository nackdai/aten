#include "ao/ao.h"
#include "kernel/StreamCompaction.h"
#include "kernel/pt_common.h"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"
#include "aten4idaten.h"

namespace idaten {
    void AORenderer::update(
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
    }

#ifdef __AT_DEBUG__
    static bool doneSetStackSize = false;
#endif

    void AORenderer::render(
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

        m_compaction.init(width * height, 1024);

        m_isects.init(width * height);
        m_rays.init(width * height);

        m_hitbools.init(width * height);
        m_hitidx.init(width * height);

        m_glimg.map();
        auto outputSurf = m_glimg.bind();

        auto vtxTexPos = m_vtxparamsPos.bind();
        auto vtxTexNml = m_vtxparamsNml.bind();

        {
            std::vector<cudaTextureObject_t> tmp;
            for (int32_t i = 0; i < m_nodeparam.size(); i++) {
                auto nodeTex = m_nodeparam[i].bind();
                tmp.push_back(nodeTex);
            }
            m_nodetex.writeFromHostToDeviceByNum(&tmp[0], (uint32_t)tmp.size());
        }

        if (!m_texRsc.empty())
        {
            std::vector<cudaTextureObject_t> tmp;
            for (int32_t i = 0; i < m_texRsc.size(); i++) {
                auto cudaTex = m_texRsc[i].bind();
                tmp.push_back(cudaTex);
            }
            m_tex.writeFromHostToDeviceByNum(&tmp[0], (uint32_t)tmp.size());
        }

        initPath(width, height);
        clearPath();

        static const int32_t rrBounce = 3;

        auto time = AT_NAME::timer::getSystemTime();

        for (int32_t i = 0; i < maxSamples; i++) {
            int32_t seed = time.milliSeconds;

            generatePath(
                width, height,
                false,
                i, maxBounce,
                seed);

            bounce = 0;

            while (bounce < maxBounce) {
                hitTest(
                    width, height,
                    bounce);

                onShadeMiss(width, height, bounce);

                m_compaction.compact(
                    m_hitidx,
                    m_hitbools,
                    nullptr);

                onShade(
                    width, height,
                    bounce, rrBounce,
                    vtxTexPos, vtxTexNml);

                bounce++;
            }
        }

        onGather(outputSurf, width, height);

        checkCudaErrors(cudaDeviceSynchronize());

        m_frame++;

        m_glimg.unbind();
        m_glimg.unmap();
    }
}
