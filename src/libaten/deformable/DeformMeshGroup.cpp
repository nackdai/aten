#include "deformable/DeformMeshGroup.h"
#include "misc/stream.h"

namespace aten
{
	bool DeformMeshGroup::read(FileInputStream* stream)
	{
		AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_desc, sizeof(m_desc)));

		// Read vertices.
		{
			m_vbs.resize(m_desc.numVB);

			std::vector<uint8_t> buf;

			for (uint32_t i = 0; i < m_desc.numVB; i++) {
				MeshVertex vtxDesc;
				AT_VRETURN_FALSE(AT_STREAM_READ(stream, &vtxDesc, sizeof(vtxDesc)));

				auto bytes = vtxDesc.numVtx * vtxDesc.sizeVtx;
				buf.resize(bytes);

				AT_VRETURN_FALSE(AT_STREAM_READ(stream, &buf[0], bytes));

				m_vbs[i].initNoVAO(
					vtxDesc.sizeVtx,
					vtxDesc.numVtx,
					0,
					&buf[0]);
			}
		}

		{
			m_meshs.resize(m_desc.numMeshSet);

			for (uint32_t i = 0; i < m_desc.numMeshSet; i++) {
				AT_VRETURN_FALSE(m_meshs[i].read(stream, m_vbs));
			}
		}

		return true;
	}
}