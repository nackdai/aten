#include "deformable/DeformPrimitives.h"
#include "deformable/Skeleton.h"
#include "misc/stream.h"

namespace aten
{
	bool DeformPrimitives::read(FileInputStream* stream)
	{
		AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_desc, sizeof(m_desc)));

		// 所属関節へのインデックス
		if (m_desc.numJoints > 0) {
			m_joints.resize(m_desc.numJoints);
			AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_joints[0], sizeof(int16_t) * m_desc.numJoints));
		}

		if (m_desc.numIdx > 0) {
			std::vector<uint32_t> indices(m_desc.numIdx);
			AT_VRETURN_FALSE(AT_STREAM_READ(stream, &indices[0], sizeof(uint32_t) * m_desc.numIdx));

			m_ib.init(m_desc.numIdx, &indices[0]);
		}
		
		return true;
	}

	void DeformPrimitives::render(
		const Skeleton& skeleton,
		IDeformMeshRenderHelper* helper)
	{
		for (uint32_t i = 0; i < m_desc.numJoints; i++) {
			auto jointIdx = m_joints[i];

			if (jointIdx < 0) {
				break;
			}

			const auto& mtxJoint = skeleton.getPoseMatrix(jointIdx);

			helper->applyMatrix(i, mtxJoint);
		}

		helper->commitChanges();

		uint32_t primNum = m_desc.numIdx / 3;

		m_ib.draw(*m_vb, aten::Primitive::Triangles, 0, primNum);
	}
}
