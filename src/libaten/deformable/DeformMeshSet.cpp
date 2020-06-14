#include <functional>

#include "deformable/DeformMeshSet.h"
#include "visualizer/atengl.h"

namespace aten
{
    static const char* attribName[] = {
        "position",
        "normal",
        "color",
        "uv",
        "tangent",
        "blendIndex",
        "blendWeight",
    };
    AT_STATICASSERT(AT_COUNTOF(attribName) == (uint32_t)MeshVertexFormat::Num);

    // 位置.
    static uint32_t setVtxAttribPos(VertexAttrib* attrib, uint32_t offset)
    {
        attrib->type = GL_FLOAT;
        attrib->num = 3;
        attrib->size = sizeof(GLfloat);
        attrib->offset = offset;

        offset += (uint32_t)MeshVertexSize::Position;

        return offset;
    }

    // 法線.
    static uint32_t setVtxAttribNormal(VertexAttrib* attrib, uint32_t offset)
    {
        attrib->type = GL_FLOAT;
        attrib->num = 3;
        attrib->size = sizeof(GLfloat);
        attrib->offset = offset;

        offset += (uint32_t)MeshVertexSize::Normal;

        return offset;
    }

    // 頂点カラー.
    static uint32_t setVtxAttribColor(VertexAttrib* attrib, uint32_t offset)
    {
        attrib->type = GL_UNSIGNED_BYTE;
        attrib->num = 4;
        attrib->size = sizeof(GLubyte);
        attrib->offset = offset;
        attrib->needNormalize = true;

        offset += (uint32_t)MeshVertexSize::Color;

        return offset;
    }

    // UV座標.
    static uint32_t setVtxAttribUV(VertexAttrib* attrib, uint32_t offset)
    {
        attrib->type = GL_FLOAT;
        attrib->num = 2;
        attrib->size = sizeof(GLfloat);
        attrib->offset = offset;

        offset += (uint32_t)MeshVertexSize::UV;

        return offset;
    }

    // 接ベクトル.
    static uint32_t setVtxAttribTangent(VertexAttrib* attrib, uint32_t offset)
    {
        attrib->type = GL_FLOAT;
        attrib->num = 3;
        attrib->size = sizeof(GLfloat);
        attrib->offset = offset;

        offset += (uint32_t)MeshVertexSize::Tangent;

        return offset;
    }

    // ブレンドウエイト.
    static uint32_t setVtxAttribBlendWeight(VertexAttrib* attrib, uint32_t offset)
    {
        attrib->type = GL_FLOAT;
        attrib->num = 4;
        attrib->size = sizeof(GLfloat);
        attrib->offset = offset;

        offset += (uint32_t)MeshVertexSize::BlendWeight;

        return offset;
    }

    // ブレンドインデックス.
    static uint32_t setVtxAttribBlendIndices(VertexAttrib* attrib, uint32_t offset)
    {
        attrib->type = GL_FLOAT;
        attrib->num = 4;
        attrib->size = sizeof(GLfloat);
        attrib->offset = offset;

        offset += (uint32_t)MeshVertexSize::BlendIndices;

        return offset;
    }

    using SetVtxAttribFunc = std::function<uint32_t(VertexAttrib*, uint32_t)>;

    bool DeformMeshSet::read(
        FileInputStream* stream,
        bool isGPUSkinning)
     {
        AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_desc, sizeof(m_desc)));

        if (m_desc.numSubset > 0) {
            if (m_desc.fmt > 0) {
                uint32_t attribNum = 0;

                for (uint32_t i = 0; i < (uint32_t)MeshVertexFormat::Num; i++) {
                    if (m_desc.fmt & (1 << i)) {
                        attribNum++;
                    }
                }

                if (attribNum > 0) {
                    m_prims.resize(m_desc.numSubset);

                    for (auto& prim : m_prims) {
                        AT_VRETURN_FALSE(prim.read(stream, isGPUSkinning));
                    }
                }
            }
        }

        return true;
    }

    void DeformMeshSet::initGLResources(
        shader* shd,
        bool isGPUSkinning,
        std::vector<GeomVertexBuffer>& vbs)
    {
        static SetVtxAttribFunc funcSetVtxAttrib[] = {
            setVtxAttribPos,
            setVtxAttribNormal,
            setVtxAttribColor,
            setVtxAttribUV,
            setVtxAttribTangent,
            setVtxAttribBlendIndices,
            setVtxAttribBlendWeight,
        };
        AT_STATICASSERT(AT_COUNTOF(funcSetVtxAttrib) == (uint32_t)MeshVertexFormat::Num);

        // 頂点バッファのアトリビュートを作成.

        uint32_t offset = 0;
        uint32_t attribNum = 0;

        VertexAttrib attribs[(uint32_t)MeshVertexFormat::Num];

        for (uint32_t i = 0; i < (uint32_t)MeshVertexFormat::Num; i++) {
            if (m_desc.fmt & (1 << i)) {
                offset = funcSetVtxAttrib[i](&attribs[attribNum], offset);
                attribs[attribNum].name = attribName[i];
                attribNum++;
            }
        }

        if (attribNum > 0) {
            for (auto& prim : m_prims) {
                if (isGPUSkinning) {
                    // Nothing...
                }
                else {
                    const auto& primDesc = prim.getDesc();
                    auto& vb = vbs[primDesc.idxVB];

                    AT_ASSERT(shd);
                    vb.createVAOByAttribName(shd, attribs, attribNum);

                    prim.setVB(&vb);
                }
            }
        }
    }

    void DeformMeshSet::setExternalVertexBuffer(GeomMultiVertexBuffer& vb)
    {
        for (auto& prim : m_prims) {
            prim.setVB(&vb);
        }
    }

    void DeformMeshSet::render(
        const context& ctxt,
        const SkeletonController& skeleton,
        IDeformMeshRenderHelper* helper,
        bool isGPUSkinning)
    {
        // material...
        helper->applyMaterial(ctxt, m_desc.mtrl);

        for (auto& prim : m_prims) {
            prim.render(skeleton, helper, isGPUSkinning);
        }
    }
}
