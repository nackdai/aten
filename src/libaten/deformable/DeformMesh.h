#pragma once

#include "deformable/DeformMeshGroup.h"
#include "deformable/SkinningVertex.h"

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

		void getGeometryData(
			std::vector<SkinningVertex>& vtx,
			std::vector<uint32_t>& idx) const;

	private:
		MeshHeader m_header;

		std::vector<DeformMeshGroup> m_groups;
	};
}
