#pragma once

#include "deformable/DeformMeshSet.h"
#include "deformable/SkinningVertex.h"

namespace aten
{
	class FileInputStream;
	class SkeletonController;

	/** メッシュグループ.
	 *
	 * LODのレベルごとのメッシュセットの集まり
	 */
	class DeformMeshGroup {
		friend class DeformMesh;

	public:
		DeformMeshGroup() {}
		~DeformMeshGroup() {}

	private:
		bool read(
			FileInputStream* stream,
			IDeformMeshReadHelper* helper,
			bool needKeepGeometryData);

		void render(
			const SkeletonController& skeleton,
			IDeformMeshRenderHelper* helper);

		void getGeometryData(
			std::vector<SkinningVertex>& vtx,
			std::vector<uint32_t>& idx) const;

	private:
		MeshGroup m_desc;

		std::vector<uint8_t> m_vertices;

		std::vector<DeformMeshSet> m_meshs;
		std::vector<GeomVertexBuffer> m_vbs;
	};
}
