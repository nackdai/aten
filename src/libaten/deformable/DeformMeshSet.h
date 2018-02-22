#pragma once

#include "deformable/DeformPrimitives.h"

namespace aten
{
	class FileInputStream;

	/** メッシュセット.
	 *
	 * マテリアルごとのプリミティブセットの集まり
	 */
	class DeformMeshSet {
		friend class DeformMeshGroup;

	public:
		DeformMeshSet() {}
		~DeformMeshSet() {}

	private:
		bool read(
			FileInputStream* stream,
			std::vector<GeomVertexBuffer>& vbs);

	private:
		MeshSet m_desc;

		std::vector<DeformPrimitives> m_prims;
	};
}
