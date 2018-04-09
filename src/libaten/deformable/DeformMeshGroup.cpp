#include "deformable/DeformMeshGroup.h"
#include "misc/stream.h"

#include <algorithm>
#include <iterator>

namespace aten
{
	bool DeformMeshGroup::read(
		FileInputStream* stream,
		IDeformMeshReadHelper* helper,
		bool needKeepGeometryData)
	{
		AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_desc, sizeof(m_desc)));

		// Read vertices.
		{
			m_vbs.resize(m_desc.numVB);

			for (uint32_t i = 0; i < m_desc.numVB; i++) {
				MeshVertex vtxDesc;
				AT_VRETURN_FALSE(AT_STREAM_READ(stream, &vtxDesc, sizeof(vtxDesc)));

				auto bytes = vtxDesc.numVtx * vtxDesc.sizeVtx;

				std::vector<uint8_t> buf;
				buf.resize(bytes);

				AT_VRETURN_FALSE(AT_STREAM_READ(stream, &buf[0], bytes));

				m_vbs[i].initNoVAO(
					vtxDesc.sizeVtx,
					vtxDesc.numVtx,
					0,
					&buf[0]);

				if (needKeepGeometryData) {
					// Need to keep vertices data.
					std::copy(
						buf.begin(),
						buf.end(),
						std::back_inserter(m_vertices));
				}
			}
		}

		{
			m_meshs.resize(m_desc.numMeshSet);

			for (uint32_t i = 0; i < m_desc.numMeshSet; i++) {
				AT_VRETURN_FALSE(m_meshs[i].read(stream, helper, m_vbs, needKeepGeometryData));
			}
		}

		return true;
	}

	void DeformMeshGroup::render(
		const SkeletonController& skeleton,
		IDeformMeshRenderHelper* helper)
	{
		for (auto& mesh : m_meshs) {
			mesh.render(skeleton, helper);
		}
	}

	void DeformMeshGroup::getGeometryData(
		std::vector<SkinningVertex>& vtx,
		std::vector<uint32_t>& idx) const
	{
		uint32_t totalVtxNum = 0;

		for (uint32_t i = 0; i < m_desc.numVB; i++) {
			totalVtxNum += m_vbs[i].getVtxNum();
		}

		vtx.reserve(totalVtxNum);

		// TODO
		// 頂点フォーマット固定...
		AT_ASSERT(sizeof(uint8_t) * m_vertices.size() == sizeof(SkinningVertex) * totalVtxNum);

		uint32_t curPos = 0;

		// Vertex.
		for (uint32_t i = 0; i < m_desc.numVB; i++) {
			auto numVtx = m_vbs[i].getVtxNum();

			// TODO
			// 頂点フォーマット固定...
			const SkinningVertex* pvtx = reinterpret_cast<const SkinningVertex*>(&m_vertices[0]);
			auto size = numVtx * sizeof(SkinningVertex);

			// Copy.
			memcpy(&vtx[0] + curPos, pvtx, size);

			curPos += numVtx;
		}

		// Index.
		for (uint32_t i = 0; i < m_desc.numMeshSet; i++) {
			const auto& prims = m_meshs[i].getPrimitives();

			for (const auto& prim : prims) {
				prim.getIndices(idx);
			}
		}
	}
}