#include "deformable/DeformMesh.h"
#include "misc/stream.h"

namespace aten
{
	bool DeformMesh::read(FileInputStream* stream)
	{
		AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_header, sizeof(m_header)));

		m_groups.resize(m_header.numMeshGroup);

		for (uint32_t i = 0; i < m_header.numMeshGroup; i++) {
			AT_VRETURN_FALSE(m_groups[i].read(stream));
		}

		return true;
	}
}
