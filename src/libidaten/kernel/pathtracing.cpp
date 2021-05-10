#include "kernel/pathtracing.h"
#include "kernel/StreamCompaction.h"
#include "kernel/pt_common.h"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"
#include "aten4idaten.h"

namespace idaten {
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

        m_sobolMatrices.init(AT_COUNTOF(sobol::Matrices::matrices));
        m_sobolMatrices.writeByNum(sobol::Matrices::matrices, m_sobolMatrices.num());

        auto& r = aten::getRandom();

        m_random.init(width * height);
        m_random.writeByNum(&r[0], width * height);

        m_paths.init(width * height);
    }

    void PathTracing::update(
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

    void PathTracing::updateMaterial(const std::vector<aten::MaterialParameter>& mtrls)
    {
        AT_ASSERT(mtrls.size() <= m_mtrlparam.num());

        if (mtrls.size() <= m_mtrlparam.num()) {
            m_mtrlparam.writeByNum(&mtrls[0], (uint32_t)mtrls.size());

            reset();
        }
    }

    void PathTracing::enableExportToGLTextures(
        GLuint gltexPosition,
        GLuint gltexNormal,
        GLuint gltexAlbedo,
        const aten::vec3& posRange)
    {
        AT_ASSERT(gltexPosition > 0);
        AT_ASSERT(gltexNormal > 0);

        if (!need_export_gl_) {
            need_export_gl_ = true;

            position_range_ = posRange;

            gl_surfaces_.resize(3);
            gl_surfaces_[0].init(gltexPosition, CudaGLRscRegisterType::WriteOnly);
            gl_surfaces_[1].init(gltexNormal, CudaGLRscRegisterType::WriteOnly);
            gl_surfaces_[2].init(gltexAlbedo, CudaGLRscRegisterType::WriteOnly);

            gl_surface_cuda_rscs_.init(3);
        }
    }

#ifdef __AT_DEBUG__
    static bool doneSetStackSize = false;
#endif

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

        m_tileDomain = tileDomain;

        int bounce = 0;

        int width = tileDomain.w;
        int height = tileDomain.h;

        m_compaction.init(width * height, 1024);

        m_isects.init(width * height);
        m_rays.init(width * height);

        m_hitbools.init(width * height);
        m_hitidx.init(width * height);

        m_shadowRays.init(width * height);

        checkCudaErrors(cudaMemset(m_paths.ptr(), 0, m_paths.bytes()));

        m_glimg.map();
        auto outputSurf = m_glimg.bind();

        auto vtxTexPos = m_vtxparamsPos.bind();
        auto vtxTexNml = m_vtxparamsNml.bind();

        {
            std::vector<cudaTextureObject_t> tmp;
            for (int i = 0; i < m_nodeparam.size(); i++) {
                auto nodeTex = m_nodeparam[i].bind();
                tmp.push_back(nodeTex);
            }
            m_nodetex.writeByNum(&tmp[0], (uint32_t)tmp.size());
        }

        if (!m_texRsc.empty())
        {
            std::vector<cudaTextureObject_t> tmp;
            for (int i = 0; i < m_texRsc.size(); i++) {
                auto cudaTex = m_texRsc[i].bind();
                tmp.push_back(cudaTex);
            }
            m_tex.writeByNum(&tmp[0], (uint32_t)tmp.size());
        }

        if (need_export_gl_) {
            std::vector<cudaSurfaceObject_t> tmp;
            for (int i = 0; i < gl_surfaces_.size(); i++) {
                gl_surfaces_[i].map();
                tmp.push_back(gl_surfaces_[i].bind());
            }
            gl_surface_cuda_rscs_.writeByNum(tmp.data(), (uint32_t)tmp.size());
        }

        static const int rrBounce = 3;

        auto time = AT_NAME::timer::getSystemTime();

        for (int i = 0; i < maxSamples; i++) {
            onGenPath(
                width, height,
                i, maxSamples,
                vtxTexPos,
                vtxTexNml);

            bounce = 0;

            while (bounce < maxBounce) {
                onHitTest(
                    width, height,
                    vtxTexPos);

                onShadeMiss(width, height, bounce);

                m_compaction.compact(
                    m_hitidx,
                    m_hitbools,
                    nullptr);

                onShade(
                    width, height,
                    i,
                    bounce, rrBounce,
                    vtxTexPos, vtxTexNml);

                bounce++;
            }
        }

        onGather(outputSurf, m_paths, width, height);

        checkCudaErrors(cudaDeviceSynchronize());

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

            for (int i = 0; i < gl_surfaces_.size(); i++) {
                gl_surfaces_[i].unbind();
                gl_surfaces_[i].unmap();
            }
        }

        m_glimg.unbind();
        m_glimg.unmap();
    }
}
