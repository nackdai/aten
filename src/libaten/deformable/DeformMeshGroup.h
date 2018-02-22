#pragma once

#include "deformable/DeformMeshSet.h"

namespace aten
{
	class FileInputStream;

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
		bool read(FileInputStream* stream);

	private:
		MeshGroup m_desc;

		std::vector<DeformMeshSet> m_meshs;
		std::vector<GeomVertexBuffer> m_vbs;
	};
}
