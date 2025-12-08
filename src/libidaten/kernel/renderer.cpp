#include "kernel/renderer.h"
#include "kernel/device_scene_context.cuh"
#include "volume/grid.h"

namespace idaten {
    Renderer::Renderer()
    {
        path_host_ = std::make_shared<AT_NAME::PathHost>();
        ctxt_host_ = std::make_shared<decltype(ctxt_host_)::element_type>();
    }

    void Renderer::UpdateSceneData(
        GLuint gltex,
        int32_t width, int32_t height,
        const aten::CameraParameter& camera,
        const aten::context& scene_ctxt,
        const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
        uint32_t advance_prim_num,
        uint32_t advance_vtx_num,
        std::function<const aten::Grid*(const aten::context&)> proxy_get_grid_from_host_scene_context/*= nullptr*/)
    {
        m_glimg.init(gltex, CudaGLRscRegisterType::ReadWrite);

        m_cam = camera;

        std::vector<aten::ObjectParameter> objs;
        std::vector<aten::mat4> mtxs;

        aten::tie(objs, mtxs) = scene_ctxt.GetObjectParametersAndMatrices();

        ctxt_host_->shapeparam.resize(objs.size());
        ctxt_host_->shapeparam.writeFromHostToDeviceByNum(objs.data(), objs.size());

        if (!mtxs.empty()) {
            ctxt_host_->mtxparams.resize(mtxs.size());
            ctxt_host_->mtxparams.writeFromHostToDeviceByNum(mtxs.data(), mtxs.size());
        }

        const auto mtrls = scene_ctxt.GetMetarialParemeters();
        ctxt_host_->mtrlparam.resize(mtrls.size());
        ctxt_host_->mtrlparam.writeFromHostToDeviceByNum(mtrls.data(), mtrls.size());

        const auto lights = scene_ctxt.GetLightParameters();
        if (!lights.empty()) {
            ctxt_host_->lightparam.resize(lights.size());
            ctxt_host_->lightparam.writeFromHostToDeviceByNum(lights.data(), lights.size());
        }

        const auto npr_lights = scene_ctxt.GetNprTargetLightParameters();
        if (!npr_lights.empty()) {
            ctxt_host_->npr_target_light_params.resize(npr_lights.size());
            ctxt_host_->npr_target_light_params.writeFromHostToDeviceByNum(npr_lights.data(), npr_lights.size());
        }

        const auto prims = scene_ctxt.GetPrimitiveParameters();
        if (prims.empty()) {
            ctxt_host_->primparams.resize(advance_prim_num);
        }
        else {
            ctxt_host_->primparams.resize(prims.size() + advance_prim_num);
            ctxt_host_->primparams.writeFromHostToDeviceByNum(prims.data(), prims.size());
        }

        ctxt_host_->nodeparam.resize(nodes.size());
        for (int32_t i = 0; i < nodes.size(); i++) {
            if (!nodes[i].empty()) {
                ctxt_host_->nodeparam[i].init(
                    reinterpret_cast<const aten::vec4*>(&nodes[i][0]),
                    sizeof(aten::GPUBvhNode) / sizeof(float4),
                    nodes[i].size());
            }
        }
        ctxt_host_->nodetex.resize(nodes.size());

        const auto vtxs_num = scene_ctxt.GetVertexNum();
        if (vtxs_num == 0) {
            ctxt_host_->vtxparamsPos.init(nullptr, 1, advance_vtx_num);
            ctxt_host_->vtxparamsNml.init(nullptr, 1, advance_vtx_num);
        }
        else {
            std::vector<aten::vec4> pos;
            std::vector<aten::vec4> nml;

            aten::tie(pos, nml) = scene_ctxt.GetExtractedPosAndNmlInVertices();

            ctxt_host_->vtxparamsPos.init(nullptr, 1, pos.size() + advance_vtx_num);
            ctxt_host_->vtxparamsNml.init(nullptr, 1, nml.size() + advance_vtx_num);

            ctxt_host_->vtxparamsPos.update(pos.data(), 1, pos.size());
            ctxt_host_->vtxparamsNml.update(nml.data(), 1, nml.size());
        }

        {
            auto tex_num = scene_ctxt.GetTextureNum();
            ctxt_host_->texRsc.reserve(tex_num);

            for (int32_t i = 0; i < tex_num; i++)
            {
                auto t = scene_ctxt.GetTexture(i);
                ctxt_host_->texRsc.emplace_back(idaten::CudaTexture());

                // TODO
#if 0
                if (envmapRsc.idx == i) {
                    m_texRsc[i].initAsMipmap(texs[i].ptr, texs[i].width, texs[i].height, 100);
                }
                else {
                    m_texRsc[i].init(texs[i].ptr, texs[i].width, texs[i].height);
                }
#else
                ctxt_host_->texRsc[i].initAsMipmap(
                    t->colors().data(),
                    t->width(), t->height(),
                    1,
                    t->GetFilterMode(), t->GetAddressMode());
#endif
            }
            ctxt_host_->tex.resize(tex_num);

            ctxt_host_->ctxt.scene_rendering_config = scene_ctxt.scene_rendering_config;
        }

        if (proxy_get_grid_from_host_scene_context) {
            const auto* grid_holder = proxy_get_grid_from_host_scene_context(scene_ctxt);
            const auto grids_num = grid_holder ? grid_holder->GetGridsNum() : 0;
            if (grids_num > 0) {
                auto* const* grids = grid_holder->GetGrids();
                ctxt_host_->grids.writeFromHostToDeviceByNum(grids, grids_num);
            }
        }
    }

    void Renderer::updateBVH(
        const aten::context& scene_ctxt,
        const std::vector<std::vector<aten::GPUBvhNode>>& nodes)
    {
        std::vector<aten::ObjectParameter> objs;
        std::vector<aten::mat4> mtxs;

        aten::tie(objs, mtxs) = scene_ctxt.GetObjectParametersAndMatrices();

        ctxt_host_->shapeparam.writeFromHostToDeviceByNum(objs.data(), objs.size());

        if (!mtxs.empty()) {
            ctxt_host_->mtxparams.writeFromHostToDeviceByNum(&mtxs[0], mtxs.size());
        }

        // Only for top layer...
        ctxt_host_->nodeparam[0].init(
            (aten::vec4*)&nodes[0][0],
            sizeof(aten::GPUBvhNode) / sizeof(float4),
            nodes[0].size());
    }

    void Renderer::updateGeometry(
        std::vector<CudaGLBuffer>& vertices,
        uint32_t vtxOffsetCount,
        TypedCudaMemory<aten::TriangleParameter>& triangles,
        uint32_t triOffsetCount)
    {
        // Vertex position.
        {
            vertices[0].map();

            aten::vec4* data = nullptr;
            size_t bytes = 0;
            vertices[0].bind((void**)&data, bytes);

            uint32_t num = (uint32_t)(bytes / sizeof(float4));

            ctxt_host_->vtxparamsPos.update(data, 1, num, vtxOffsetCount);

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

            ctxt_host_->vtxparamsNml.update(data, 1, num, vtxOffsetCount);

            vertices[1].unbind();
            vertices[1].unmap();
        }

        // Triangles.
        {
            auto size = triangles.bytes();
            auto offset = triOffsetCount * triangles.stride();

            ctxt_host_->primparams.writeFromHostToDeviceByBytes(triangles.data(), size, offset);
        }
    }

    void Renderer::updateCamera(const aten::CameraParameter& camera)
    {
        m_cam = camera;
    }

    void Renderer::updateMaterial(const std::vector<aten::MaterialParameter>& mtrls)
    {
        AT_ASSERT(mtrls.size() <= ctxt_host_->mtrlparam.num());

        if (mtrls.size() <= ctxt_host_->mtrlparam.num()) {
            ctxt_host_->mtrlparam.writeFromHostToDeviceByNum(&mtrls[0], (uint32_t)mtrls.size());
            reset();
        }
    }

    void Renderer::updateLight(const aten::context& scene_ctxt, bool is_npr_target_light)
    {
        const auto lights = is_npr_target_light
            ? scene_ctxt.GetNprTargetLightParameters()
            : scene_ctxt.GetLightParameters();

        AT_ASSERT(lights.size() <= ctxt_host_->lightparam.num());

        if (lights.size() <= ctxt_host_->lightparam.num()) {
            if (is_npr_target_light) {
                ctxt_host_->npr_target_light_params.writeFromHostToDeviceByNum(&lights[0], (uint32_t)lights.size());
            }
            else {
                ctxt_host_->lightparam.writeFromHostToDeviceByNum(&lights[0], (uint32_t)lights.size());
            }
            reset();
        }
    }

    void Renderer::UpdateTexture(int32_t idx, const aten::context& ctxt)
    {
        AT_ASSERT(idx < ctxt_host_->texRsc.size());
        if (idx < ctxt_host_->texRsc.size()) {
            auto host_tex = ctxt.GetTexture(idx);
            const auto pixels = host_tex->width() * host_tex->height();
            ctxt_host_->texRsc[idx].CopyFromHost(
                host_tex->colors().data(),
                host_tex->width(), host_tex->height()
            );
        }
    }

    void Renderer::UpdateSceneRenderingConfig(const aten::context& ctxt)
    {
        ctxt_host_->ctxt.scene_rendering_config = ctxt.scene_rendering_config;
    }

    uint32_t Renderer::getRegisteredTextureNum() const
    {
        return static_cast<uint32_t>(ctxt_host_->texRsc.size());
    }

    std::vector<idaten::CudaTextureResource>& Renderer::getCudaTextureResourceForBvhNodes()
    {
        return ctxt_host_->nodeparam;
    }

    idaten::CudaTextureResource Renderer::getCudaTextureResourceForVtxPos()
    {
        return ctxt_host_->vtxparamsPos;
    }

    idaten::StreamCompaction& Renderer::getCompaction()
    {
        return m_compaction;
    }
}
