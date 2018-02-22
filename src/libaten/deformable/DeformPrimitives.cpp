#include "deformable/DeformPrimitives.h"
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
}
