#include "svgf/svgf.h"

#include "kernel/StreamCompaction.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

//#pragma optimize( "", off)

namespace idaten
{
    void SVGFPathTracing::update(
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

        m_sobolMatrices.init(AT_COUNTOF(sobol::Matrices::matrices));
        m_sobolMatrices.writeByNum(sobol::Matrices::matrices, m_sobolMatrices.num());

        auto& r = aten::getRandom();
        m_random.init(width * height);
        m_random.writeByNum(&r[0], width * height);

        for (int i = 0; i < 2; i++) {
            m_aovNormalDepth[i].init(width * height);
            m_aovTexclrMeshid[i].init(width * height);
            m_aovColorVariance[i].init(width * height);
            m_aovMomentTemporalWeight[i].init(width * height);
        }

        for (int i = 0; i < AT_COUNTOF(m_atrousClrVar); i++) {
            m_atrousClrVar[i].init(width * height);
        }

        m_tmpBuf.init(width * height);
    }

    void SVGFPathTracing::updateBVH(
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

    void SVGFPathTracing::updateGeometry(
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

    void SVGFPathTracing::setGBuffer(
        GLuint gltexGbuffer,
        GLuint gltexMotionDepthbuffer)
    {
        m_gbuffer.init(gltexGbuffer, idaten::CudaGLRscRegisterType::ReadOnly);
        m_motionDepthBuffer.init(gltexMotionDepthbuffer, idaten::CudaGLRscRegisterType::ReadOnly);
    }

    static bool doneSetStackSize = false;

    void SVGFPathTracing::render(
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

        m_shadowRays.init(width * height * ShadowRayNum);

        onInit(width, height);

        CudaGLResourceMapper rscmap(&m_glimg);
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

        if (width > 1280 || height > 720) {
            int w = (width + 1) / 2;
            int h = (height + 1) / 2;

            int compactionW = std::max(w, width - w);
            int compactionH = std::max(h, height - h);

            m_hitbools.init(compactionW * compactionH);
            m_hitidx.init(compactionW * compactionH);

            m_compaction.init(
                compactionW * compactionH,
                1024);

            for (int nx = 0; nx < 2; nx++) {
                for (int ny = 0; ny < 2; ny++) {
                    int x = nx * w;
                    int y = ny * h;

                    onClear();

                    onRender(
                        TileDomain(x, y, w, h),
                        width, height, maxSamples, maxBounce,
                        outputSurf,
                        vtxTexPos,
                        vtxTexNml);
                }
            }

#if 1
            for (int nx = 0; nx < 2; nx++) {
                for (int ny = 0; ny < 2; ny++) {
                    int x = nx * w;
                    int y = ny * h;

                    onDenoise(
                        TileDomain(x, y, w, h),
                        width, height,
                        outputSurf);
                }
            }

            if (m_mode == Mode::SVGF) {
                static const int ITER = 5;

                for (int i = 0; i < ITER; i++) {
                    for (int nx = 0; nx < 2; nx++) {
                        for (int ny = 0; ny < 2; ny++) {
                            int x = nx * w;
                            int y = ny * h;

                            m_tileDomain = TileDomain(x, y, w, h);

                            onAtrousFilterIter(
                                i, ITER,
                                outputSurf,
                                width, height);
                        }
                    }
                }

                onCopyFromTmpBufferToAov(width, height);
            }
#endif
        }
        else {
            m_hitbools.init(width * height);
            m_hitidx.init(width * height);

            m_compaction.init(
                width * height,
                1024);

            onClear();

            TileDomain tileDomain(0, 0, width, height);

            onRender(
                tileDomain,
                width, height, maxSamples, maxBounce,
                outputSurf,
                vtxTexPos,
                vtxTexNml);

            onDenoise(
                tileDomain,
                width, height,
                outputSurf);

            if (m_mode == Mode::SVGF)
            {
                onAtrousFilter(outputSurf, width, height);
                onCopyFromTmpBufferToAov(width, height);
            }
        }

        {
            m_mtxPrevW2V = m_mtxW2V;

            pick(
                m_pickedInfo.ix, m_pickedInfo.iy,
                width, height,
                vtxTexPos);

            //checkCudaErrors(cudaDeviceSynchronize());

            // Toggle aov buffer pos.
            m_curAOVPos = 1 - m_curAOVPos;

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

    void SVGFPathTracing::onRender(
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

            onGenPath(
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

                onShadeMiss(width, height, bounce);

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
            onDisplayAOV(outputSurf, width, height, vtxTexPos);
        }
        else {
            if (isFirstFrame()) {
                onGather(outputSurf, width, height, maxSamples);
            }
            else {
                onCopyBufferForTile(width, height);
            }
        }
    }

    void SVGFPathTracing::onDenoise(
        const TileDomain& tileDomain,
        int width, int height,
        cudaSurfaceObject_t outputSurf)
    {
        m_tileDomain = tileDomain;

        if (m_mode == Mode::SVGF
            || m_mode == Mode::TF
            || m_mode == Mode::VAR)
        {
            if (isFirstFrame()) {
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

    void SVGFPathTracing::setStream(cudaStream_t stream)
    {
        m_stream = stream;
        m_compaction.setStream(stream);
    }
}
