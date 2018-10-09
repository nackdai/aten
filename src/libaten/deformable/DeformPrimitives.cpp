#include <algorithm>
#include <iterator>

#include "deformable/DeformPrimitives.h"

namespace aten
{
    bool DeformPrimitives::read(
        FileInputStream* stream,
        bool isGPUSkinning)
    {
        AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_desc, sizeof(m_desc)));

        // 所属関節へのインデックス
        if (m_desc.numJoints > 0) {
            m_joints.resize(m_desc.numJoints);
            AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_joints[0], sizeof(int16_t) * m_desc.numJoints));
        }

        if (m_desc.numIdx > 0) {
            m_indices.resize(m_desc.numIdx);
            AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_indices[0], sizeof(uint32_t) * m_desc.numIdx));

            m_ib.init(m_desc.numIdx, &m_indices[0]);
        }

        if (!isGPUSkinning) {
            // Not need to keep indices data.
            m_indices.clear();
        }
        
        return true;
    }

    void DeformPrimitives::render(
        const SkeletonController& skeleton,
        IDeformMeshRenderHelper* helper,
        bool isGPUSkinning)
    {
        if (isGPUSkinning) {
            // Nothing...
        }
        else {
            for (uint32_t i = 0; i < m_desc.numJoints; i++) {
                auto jointIdx = m_joints[i];

                if (jointIdx < 0) {
                    break;
                }

                const auto& mtxJoint = skeleton.getMatrix(jointIdx);

                helper->applyMatrix(i, mtxJoint);
            }
        }

        helper->commitChanges(isGPUSkinning, m_triOffset);

        uint32_t primNum = m_desc.numIdx / 3;

        m_ib.draw(*m_vb, aten::Primitive::Triangles, 0, primNum);
    }

    void DeformPrimitives::getIndices(std::vector<uint32_t>& indices) const
    {
        AT_ASSERT(!m_indices.empty());

        std::copy(
            m_indices.begin(),
            m_indices.end(),
            std::back_inserter(indices));
    }
}
