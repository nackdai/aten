#pragma once

#include "deformable/DeformMeshGroup.h"

namespace aten
{
	class FileInputStream;
	class Skeleton;

	/** メッシュデータ.
	 */
	class DeformMesh {
		friend class deformable;

	private:
		DeformMesh() {}
		~DeformMesh() {}

	private:
		bool read(
			FileInputStream* stream,
			IDeformMeshReadHelper* helper);

		void render(
			const Skeleton& skeleton,
			IDeformMeshRenderHelper* helper);

	private:
		MeshHeader m_header;

		std::vector<DeformMeshGroup> m_groups;
	};
}
