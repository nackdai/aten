#pragma once

#include "deformable/DeformMeshGroup.h"

namespace aten
{
	class FileInputStream;

	/** メッシュデータ.
	 */
	class DeformMesh {
	private:
		DeformMesh() {}
		~DeformMesh() {}

	private:
		bool read(FileInputStream* stream);

	private:
		MeshHeader m_header;

		std::vector<DeformMeshGroup> m_groups;
	};
}
