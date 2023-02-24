#include "deformable/DeformMesh.h"

namespace aten
{
    bool DeformMesh::read(FileInputStream* stream)
    {
        AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_header, sizeof(m_header)));

        // TODO
        // Not support multi group (aka. LOD).
        m_header.numMeshGroup = 1;

        bool isGPUSkinning = m_header.isGPUSkinning;

        m_groups.resize(m_header.numMeshGroup);

        for (uint32_t i = 0; i < m_header.numMeshGroup; i++) {
            AT_VRETURN_FALSE(m_groups[i].read(stream, isGPUSkinning));
        }

        return true;
    }

    void DeformMesh::initGLResources(shader* shd)
    {
        bool isGPUSkinning = m_header.isGPUSkinning;

        for (uint32_t i = 0; i < m_header.numMeshGroup; i++) {
            m_groups[i].initGLResources(shd, isGPUSkinning);
        }
    }

    void DeformMesh::render(
        const context& ctxt,
        const SkeletonController& skeleton,
        IDeformMeshRenderHelper* helper)
    {
        bool isGPUSkinning = m_header.isGPUSkinning;
        m_groups[0].render(ctxt, skeleton, helper, isGPUSkinning);
    }

    void DeformMesh::getGeometryData(
        const context& ctxt,
        std::vector<SkinningVertex>& vtx,
        std::vector<uint32_t>& idx,
        std::vector<aten::TriangleParameter>& tris) const
    {
        m_groups[0].getGeometryData(ctxt, vtx, idx, tris);
    }
}
