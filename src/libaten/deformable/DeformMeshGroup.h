#pragma once

#include "deformable/DeformMeshSet.h"

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
			IDeformMeshReadHelper* helper);

		void render(
			const SkeletonController& skeleton,
			IDeformMeshRenderHelper* helper);

	private:
		MeshGroup m_desc;

		std::vector<DeformMeshSet> m_meshs;
		std::vector<GeomVertexBuffer> m_vbs;
	};
}
