#include "kernel/pathtracing.h"

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
    void PathTracing::update(
        GLuint gltex,
        int width, int height,
        const aten::CameraParameter& camera,
        const std::vector<aten::GeomParameter>& shapes,
        const std::vector<aten::MaterialParameter>& mtrls,
        const std::vector<aten::LightParameter>& lights,
        const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
        const std::vector<aten::PrimitiveParamter>& prims,
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

        aov_position_.init(width * height);
        aov_nml_.init(width * height);
        aov_albedo_.init(width * height);
    }

    void PathTracing::updateBVH(
        const std::vector<aten::GeomParameter>& geoms,
        const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
        const std::vector<aten::mat4>& mtxs)
    {
        m_shapeparam.writeByNum(&geoms[0], geoms.size());

        // Only for top layer...
        m_nodeparam[0].init(
            (aten::vec4*)&nodes[0][0],
            sizeof(aten::GPUBvhNode) / sizeof(float4),
            nodes[0].size());

        if (!mtxs.empty()) {
            m_mtxparams.writeByNum(&mtxs[0], mtxs.size());
        }
    }

    void PathTracing::updateMaterial(const std::vector<aten::MaterialParameter>& mtrls)
    {
        AT_ASSERT(mtrls.size() <= m_mtrlparam.num());

        if (mtrls.size() <= m_mtrlparam.num()) {
            m_mtrlparam.writeByNum(&mtrls[0], (uint32_t)mtrls.size());
            reset();
        }
    }

    void PathTracing::updateLight(const std::vector<aten::LightParameter>& lights)
    {
        AT_ASSERT(lights.size() <= m_lightparam.num());

        if (lights.size() <= m_lightparam.num()) {
            m_lightparam.writeByNum(&lights[0], (uint32_t)lights.size());
            reset();
        }
    }

    void PathTracing::updateGeometry(
        std::vector<CudaGLBuffer>& vertices,
        uint32_t vtxOffsetCount,
        TypedCudaMemory<aten::PrimitiveParamter>& triangles,
        uint32_t triOffsetCount)
    {
        // Vertex position.
        {
            vertices[0].map();

            aten::vec4* data = nullptr;
            size_t bytes = 0;
            vertices[0].bind((void**)&data, bytes);

            uint32_t num = (uint32_t)(bytes / sizeof(float4));

            m_vtxparamsPos.update(data, 1, num, vtxOffsetCount);

            vertices[0].unbind();
            vertices[0].unmap();
        }

        // Vertex normal.
        {
            vertices[1].map();

            aten::vec4* data = nullptr;
            size_t bytes = 0;
            vertices[1].bind((void**)&data, bytes);

            uint32_t num = (uint32_t)(bytes / sizeof(float4));

            m_vtxparamsNml.update(data, 1, num, vtxOffsetCount);

            vertices[1].unbind();
            vertices[1].unmap();
        }

        // Triangles.
        {
            auto size = triangles.bytes();
            auto offset = triOffsetCount * triangles.stride();

            m_primparams.write(triangles.ptr(), size, offset);
        }
    }

    static bool doneSetStackSize = false;

    void PathTracing::render(
        const TileDomain& tileDomain,
        int maxSamples,
        int maxBounce)
    {
#ifdef __AT_DEBUG__
        if (!doneSetStackSize) {
            size_t val = 0;
            cudaThreadGetLimit(&val, cudaLimitStackSize);
            cudaThreadSetLimit(cudaLimitStackSize, val * 4);
            doneSetStackSize = true;
        }
#endif

        int bounce = 0;

        int width = tileDomain.w;
        int height = tileDomain.h;

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
                for (int i = 0; i < m_nodeparam.size(); i++) {
                    auto nodeTex = m_nodeparam[i].bind();
                    tmp.push_back(nodeTex);
                }
                m_nodetex.writeByNum(&tmp[0], tmp.size());
            }

            if (!m_texRsc.empty())
            {
                std::vector<cudaTextureObject_t> tmp;
                for (int i = 0; i < m_texRsc.size(); i++) {
                    auto cudaTex = m_texRsc[i].bind();
                    tmp.push_back(cudaTex);
                }
                m_tex.writeByNum(&tmp[0], tmp.size());
            }

            m_isListedTextureObject = true;
        }
        else {
            for (int i = 0; i < m_nodeparam.size(); i++) {
                auto nodeTex = m_nodeparam[i].bind();
            }
            for (int i = 0; i < m_texRsc.size(); i++) {
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
            m_frame++;

            {
                m_vtxparamsPos.unbind();
                m_vtxparamsNml.unbind();

                for (int i = 0; i < m_nodeparam.size(); i++) {
                    m_nodeparam[i].unbind();
                }

                for (int i = 0; i < m_texRsc.size(); i++) {
                    m_texRsc[i].unbind();
                }
            }
        }
    }

    void PathTracing::onRender(
        const TileDomain& tileDomain,
        int width, int height,
        int maxSamples,
        int maxBounce,
        cudaSurfaceObject_t outputSurf,
        cudaTextureObject_t vtxTexPos,
        cudaTextureObject_t vtxTexNml)
    {
        m_tileDomain = tileDomain;

        static const int rrBounce = 3;

        // Set bounce count to 1 forcibly, aov render mode.
        maxBounce = (m_mode == Mode::AOVar ? 1 : maxBounce);

        auto time = AT_NAME::timer::getSystemTime();

        for (int i = 0; i < maxSamples; i++) {
            int seed = time.milliSeconds;
            //int seed = 0;

            generatePath(
                m_mode == Mode::AOVar,
                i, maxBounce,
                seed,
                vtxTexPos,
                vtxTexNml);

            int bounce = 0;

            while (bounce < maxBounce) {
                onHitTest(
                    width, height,
                    bounce,
                    vtxTexPos);

                missShade(width, height, bounce);

                int hitcount = 0;
                m_compaction.compact(
                    m_hitidx,
                    m_hitbools);

                //AT_PRINTF("%d\n", hitcount);

                onShade(
                    outputSurf,
                    width, height,
                    i,
                    bounce, rrBounce,
                    vtxTexPos, vtxTexNml);

                bounce++;
            }
        }

        if (m_mode == Mode::PT) {
            onGather(outputSurf, width, height, maxSamples);
        }
        else if (m_mode == Mode::AOVar) {
        }
        else {
            AT_ASSERT(false);
        }
    }

    void PathTracing::setStream(cudaStream_t stream)
    {
        m_stream = stream;
        m_compaction.setStream(stream);
    }
}
