#include "kernel/renderer.h"

namespace idaten {
    void Renderer::update(
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
#if 0
        size_t size_stack = 0;
        checkCudaErrors(cudaThreadGetLimit(&size_stack, cudaLimitStackSize));
        checkCudaErrors(cudaThreadSetLimit(cudaLimitStackSize, 12928));
        checkCudaErrors(cudaThreadGetLimit(&size_stack, cudaLimitStackSize));

        AT_PRINTF("Stack size %d\n", size_stack);
#endif

        m_glimg.init(gltex, CudaGLRscRegisterType::ReadWrite);

        m_cam = camera;

        ctxt_host_.shapeparam.resize(shapes.size());
        ctxt_host_.shapeparam.writeFromHostToDeviceByNum(&shapes[0], shapes.size());

        ctxt_host_.mtrlparam.resize(mtrls.size());
        ctxt_host_.mtrlparam.writeFromHostToDeviceByNum(&mtrls[0], mtrls.size());

        AT_ASSERT_LOG(!lights.empty(), "No Lights!!");

        if (!lights.empty()) {
            ctxt_host_.lightparam.resize(lights.size());
            ctxt_host_.lightparam.writeFromHostToDeviceByNum(&lights[0], lights.size());
        }

        if (prims.empty()) {
            ctxt_host_.primparams.resize(advancePrimNum);
        }
        else {
            ctxt_host_.primparams.resize(prims.size() + advancePrimNum);
            ctxt_host_.primparams.writeFromHostToDeviceByNum(&prims[0], prims.size());
        }

        if (!mtxs.empty()) {
            ctxt_host_.mtxparams.resize(mtxs.size());
            ctxt_host_.mtxparams.writeFromHostToDeviceByNum(&mtxs[0], mtxs.size());
        }

        ctxt_host_.nodeparam.resize(nodes.size());
        for (int32_t i = 0; i < nodes.size(); i++) {
            if (!nodes[i].empty()) {
                ctxt_host_.nodeparam[i].init(
                    (aten::vec4*)&nodes[i][0],
                    sizeof(aten::GPUBvhNode) / sizeof(float4),
                    nodes[i].size());
            }
        }
        ctxt_host_.nodetex.resize(nodes.size());

        if (vtxs.empty()) {
            ctxt_host_.vtxparamsPos.init(nullptr, 1, advanceVtxNum);
            ctxt_host_.vtxparamsNml.init(nullptr, 1, advanceVtxNum);
        }
        else {
            // TODO
            std::vector<aten::vec4> pos;
            std::vector<aten::vec4> nml;

            for (const auto& v : vtxs) {
                pos.push_back(aten::vec4(v.pos.x, v.pos.y, v.pos.z, v.uv.x));
                nml.push_back(aten::vec4(v.nml.x, v.nml.y, v.nml.z, v.uv.y));
            }

#if 0
            m_vtxparamsPos.init((aten::vec4*)&pos[0], 1, pos.size() + advanceVtxNum);
            m_vtxparamsNml.init((aten::vec4*)&nml[0], 1, nml.size() + advanceVtxNum);
#else
            ctxt_host_.vtxparamsPos.init(nullptr, 1, pos.size() + advanceVtxNum);
            ctxt_host_.vtxparamsNml.init(nullptr, 1, nml.size() + advanceVtxNum);

            ctxt_host_.vtxparamsPos.update(&pos[0], 1, pos.size());
            ctxt_host_.vtxparamsNml.update(&nml[0], 1, nml.size());
#endif
        }

        if (!texs.empty()) {
            for (int32_t i = 0; i < texs.size(); i++) {
                ctxt_host_.texRsc.push_back(idaten::CudaTexture());

                // TODO
#if 0
                if (envmapRsc.idx == i) {
                    m_texRsc[i].initAsMipmap(texs[i].ptr, texs[i].width, texs[i].height, 100);
                }
                else {
                    m_texRsc[i].init(texs[i].ptr, texs[i].width, texs[i].height);
                }
#else
                ctxt_host_.texRsc[i].initAsMipmap(texs[i].ptr, texs[i].width, texs[i].height, 100);
#endif
            }
            ctxt_host_.tex.resize(texs.size());
        }

        bg_ = bg_resource;
    }

    void Renderer::UpdateSceneData(
        GLuint gltex,
        int32_t width, int32_t height,
        const aten::CameraParameter& camera,
        const aten::context& scene_ctxt,
        const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
        uint32_t advance_prim_num,
        uint32_t advance_vtx_num,
        const aten::BackgroundResource& bg_resource)
    {
        m_glimg.init(gltex, CudaGLRscRegisterType::ReadWrite);

        m_cam = camera;

        std::vector<aten::ObjectParameter> objs;
        std::vector<aten::mat4> mtxs;

        aten::tie(objs, mtxs) = scene_ctxt.GetObjectParametersAndMatrices();

        ctxt_host_.shapeparam.resize(objs.size());
        ctxt_host_.shapeparam.writeFromHostToDeviceByNum(objs.data(), objs.size());

        if (!mtxs.empty()) {
            ctxt_host_.mtxparams.resize(mtxs.size());
            ctxt_host_.mtxparams.writeFromHostToDeviceByNum(mtxs.data(), mtxs.size());
        }

        const auto mtrls = scene_ctxt.GetMetarialParemeters();
        ctxt_host_.mtrlparam.resize(mtrls.size());
        ctxt_host_.mtrlparam.writeFromHostToDeviceByNum(mtrls.data(), mtrls.size());

        const auto lights = scene_ctxt.GetLightParameters();
        if (!lights.empty()) {
            ctxt_host_.lightparam.resize(lights.size());
            ctxt_host_.lightparam.writeFromHostToDeviceByNum(lights.data(), lights.size());
        }

        const auto prims = scene_ctxt.GetPrimitiveParameters();
        if (prims.empty()) {
            ctxt_host_.primparams.resize(advance_prim_num);
        }
        else {
            ctxt_host_.primparams.resize(prims.size() + advance_prim_num);
            ctxt_host_.primparams.writeFromHostToDeviceByNum(prims.data(), prims.size());
        }

        ctxt_host_.nodeparam.resize(nodes.size());
        for (int32_t i = 0; i < nodes.size(); i++) {
            if (!nodes[i].empty()) {
                ctxt_host_.nodeparam[i].init(
                    reinterpret_cast<const aten::vec4*>(&nodes[i][0]),
                    sizeof(aten::GPUBvhNode) / sizeof(float4),
                    nodes[i].size());
            }
        }
        ctxt_host_.nodetex.resize(nodes.size());

        const auto vtxs_num = scene_ctxt.GetVertexNum();
        if (vtxs_num == 0) {
            ctxt_host_.vtxparamsPos.init(nullptr, 1, advance_vtx_num);
            ctxt_host_.vtxparamsNml.init(nullptr, 1, advance_vtx_num);
        }
        else {
            std::vector<aten::vec4> pos;
            std::vector<aten::vec4> nml;

            aten::tie(pos, nml) = scene_ctxt.GetExtractedPosAndNmlInVertices();

            ctxt_host_.vtxparamsPos.init(nullptr, 1, pos.size() + advance_vtx_num);
            ctxt_host_.vtxparamsNml.init(nullptr, 1, nml.size() + advance_vtx_num);

            ctxt_host_.vtxparamsPos.update(pos.data(), 1, pos.size());
            ctxt_host_.vtxparamsNml.update(nml.data(), 1, nml.size());
        }

        {
            auto tex_num = scene_ctxt.GetTextureNum();
            ctxt_host_.texRsc.reserve(tex_num);

            for (int32_t i = 0; i < tex_num; i++)
            {
                auto t = scene_ctxt.GetTexture(i);
                ctxt_host_.texRsc.emplace_back(idaten::CudaTexture());

                // TODO
#if 0
                if (envmapRsc.idx == i) {
                    m_texRsc[i].initAsMipmap(texs[i].ptr, texs[i].width, texs[i].height, 100);
                }
                else {
                    m_texRsc[i].init(texs[i].ptr, texs[i].width, texs[i].height);
                }
#else
                ctxt_host_.texRsc[i].initAsMipmap(t->colors(), t->width(), t->height(), 100);
#endif
            }
            ctxt_host_.tex.resize(tex_num);
        }

        bg_ = bg_resource;
    }

    void Renderer::updateBVH(
        const aten::context& scene_ctxt,
        const std::vector<std::vector<aten::GPUBvhNode>>& nodes)
    {
        std::vector<aten::ObjectParameter> objs;
        std::vector<aten::mat4> mtxs;

        aten::tie(objs, mtxs) = scene_ctxt.GetObjectParametersAndMatrices();

        ctxt_host_.shapeparam.writeFromHostToDeviceByNum(objs.data(), objs.size());

        if (!mtxs.empty()) {
            ctxt_host_.mtxparams.writeFromHostToDeviceByNum(&mtxs[0], mtxs.size());
        }

        // Only for top layer...
        ctxt_host_.nodeparam[0].init(
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

            ctxt_host_.vtxparamsPos.update(data, 1, num, vtxOffsetCount);

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

            ctxt_host_.vtxparamsNml.update(data, 1, num, vtxOffsetCount);

            vertices[1].unbind();
            vertices[1].unmap();
        }

        // Triangles.
        {
            auto size = triangles.bytes();
            auto offset = triOffsetCount * triangles.stride();

            ctxt_host_.primparams.writeFromHostToDeviceByBytes(triangles.data(), size, offset);
        }
    }

    void Renderer::updateCamera(const aten::CameraParameter& camera)
    {
        m_cam = camera;
    }
}
