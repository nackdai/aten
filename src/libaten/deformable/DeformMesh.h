#pragma once

#include "deformable/DeformMeshGroup.h"

namespace aten
{
	class FileInputStream;
	class SkeletonController;

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
			const SkeletonController& skeleton,
			IDeformMeshRenderHelper* helper);

		void release()
		{
			m_groups.clear();
		}

	private:
		MeshHeader m_header;

		std::vector<DeformMeshGroup> m_groups;
	};
}
