#include <algorithm>
#include <iterator>

#include "deformable/DeformMeshGroup.h"
#include "geometry/vertex.h"
#include "visualizer/atengl.h"

namespace aten
{
    bool DeformMeshGroup::read(
        FileInputStream* stream,
        bool isGPUSkinning)
    {
        AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_desc, sizeof(m_desc)));

        // Read vertices.
        {
            m_vbs.resize(m_desc.numVB);

            m_vtxTotalNum = 0;

            for (uint32_t i = 0; i < m_desc.numVB; i++) {
                MeshVertex vtxDesc;
                AT_VRETURN_FALSE(AT_STREAM_READ(stream, &vtxDesc, sizeof(vtxDesc)));

                auto bytes = vtxDesc.numVtx * vtxDesc.sizeVtx;

                m_vtxTotalNum += vtxDesc.numVtx;

                std::vector<uint8_t> buf;
                buf.resize(bytes);

                AT_VRETURN_FALSE(AT_STREAM_READ(stream, &buf[0], bytes));

                if (isGPUSkinning) {
                    // Need to keep vertices data.
                    std::copy(
                        buf.begin(),
                        buf.end(),
                        std::back_inserter(vertices_));
                }
                else {
                    m_vbs[i].initNoVAO(
                        vtxDesc.sizeVtx,
                        vtxDesc.numVtx,
                        0,
                        &buf[0]);
                }
            }
        }

        // Read mesh data..
        {
            m_meshs.resize(m_desc.numMeshSet);

            triangles_ = 0;

            for (uint32_t i = 0; i < m_desc.numMeshSet; i++) {
                AT_VRETURN_FALSE(m_meshs[i].read(stream, isGPUSkinning));

                for (auto& prim : m_meshs[i].m_prims) {
                    prim.setTriOffset(triangles_);

                    AT_ASSERT(prim.m_desc.numIdx % 3 == 0);
                    auto triNum = prim.m_desc.numIdx / 3;
                    triangles_ += triNum;
                }
            }
        }

        return true;
    }

    void DeformMeshGroup::initGLResources(
        shader* shd,
        bool isGPUSkinning)
    {
        for (uint32_t i = 0; i < m_desc.numMeshSet; i++) {
            m_meshs[i].initGLResources(shd, isGPUSkinning, m_vbs);
        }

        if (m_vbForGPUSkinning.isInitialized()) {
            return;
        }

        if (isGPUSkinning)
        {
            // TODO
            // シェーダの input とどう合わせるか...

            // NOTE
            // VBOs will be created for each attributes.
            // i.e. each attributes are dependent.
            // Variable "offset" is not used, so offset is zero.
            static const VertexAttrib attribs[] = {
                VertexAttrib(GL_FLOAT, 4, sizeof(GLfloat), 0),    // Position.
                VertexAttrib(GL_FLOAT, 4, sizeof(GLfloat), 0),    // Normal.
                VertexAttrib(GL_FLOAT, 4, sizeof(GLfloat), 0),    // Previous Position.
            };

            m_vbForGPUSkinning.init(
                m_vtxTotalNum,
                attribs,
                AT_COUNTOF(attribs),
                nullptr,
                true);

            for (auto& m : m_meshs) {
                m.setExternalVertexBuffer(m_vbForGPUSkinning);
            }
        }
    }

    void DeformMeshGroup::render(
        const context& ctxt,
        const SkeletonController& skeleton,
        IDeformMeshRenderHelper* helper,
        bool isGPUSkinning)
    {
        for (auto& mesh : m_meshs) {
            mesh.render(ctxt, skeleton, helper, isGPUSkinning);
        }
    }

    void DeformMeshGroup::getGeometryData(
        const context& ctxt,
        std::vector<SkinningVertex>& vtx,
        std::vector<uint32_t>& idx,
        std::vector<aten::TriangleParameter>& tris) const
    {
        // TODO
        // 頂点フォーマット固定...
        AT_ASSERT(sizeof(uint8_t) * vertices_.size() == sizeof(SkinningVertex) * m_vtxTotalNum);

        // Vertex.
        {
            vtx.resize(m_vtxTotalNum);

            auto size = m_vtxTotalNum * sizeof(SkinningVertex);
            memcpy(&vtx[0], &vertices_[0], size);
        }

        // Index.
        for (uint32_t i = 0; i < m_desc.numMeshSet; i++)
        {
            const auto& mtrlDesc = m_meshs[i].getDesc().mtrl;

            int32_t mtrlId = ctxt.FindMaterialIdxByName(mtrlDesc.name);

            int32_t geomId = m_meshs[i].get_mesh_id();

            const auto& prims = m_meshs[i].getPrimitives();

            const auto begin = idx.size();

            for (const auto& prim : prims) {
                prim.getIndices(idx);
            }

            const auto end = idx.size();
            AT_ASSERT((end - begin) % 3 == 0);

            for (uint32_t n = begin; n < end; n += 3) {
                tris.push_back(aten::TriangleParameter());
                auto& tri = tris.back();

                tri.idx[0] = idx[n + 0];
                tri.idx[1] = idx[n + 1];
                tri.idx[2] = idx[n + 2];

                tri.mtrlid = mtrlId;
                tri.mesh_id = geomId;

                // TODO
                tri.needNormal = 0;
            }
        }
    }
}
