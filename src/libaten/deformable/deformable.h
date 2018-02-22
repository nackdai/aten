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

	//////////////////////////////////////////////////////////////

	// For debug.
	class DeformableRenderer {
	private:
		DeformableRenderer();
		~DeformableRenderer();

	public:
		static bool init(
			int width, int height,
			const char* pathVS,
			const char* pathFS);

		static void render(deformable* mdl);

	private:
		static shader s_shd;
	};
}
