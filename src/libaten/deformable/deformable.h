#pragma once

#include "deformable/MDLFormat.h"
#include "deformable/DeformMesh.h"
#include "deformable/Skeleton.h"

namespace aten
{
	class FileInputStream;
	class shader;

	/** メッシュデータ.
	 */
	class deformable {
	public:
		deformable() {}
		~deformable() {}

	public:
		bool read(FileInputStream* stream);

		void render(shader* shd);

	private:
		DeformMesh m_mesh;
		Skeleton m_skl;
	};
}
