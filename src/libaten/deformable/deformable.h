#pragma once

#include "deformable/MDLFormat.h"
#include "deformable/DeformMesh.h"
#include "deformable/Skeleton.h"

namespace aten
{
	class shader;

	/** メッシュデータ.
	 */
	class deformable {
	public:
		deformable() {}
		~deformable() {}

	public:
		bool read(const char* path);

		void render(shader* shd);

	private:
		DeformMesh m_mesh;
		Skeleton m_skl;
	};

	//////////////////////////////////////////////////////////////

	class camera;
	class DeformMeshReadHelper;

	// For debug.
	class DeformableRenderer {
		friend class deformable;

	private:
		DeformableRenderer();
		~DeformableRenderer();

	public:
		static bool init(
			int width, int height,
			const char* pathVS,
			const char* pathFS);

		static void render(
			const camera* cam,
			deformable* mdl);

	private:
		static void initDeformMeshReadHelper(DeformMeshReadHelper* helper);

	private:
		static shader s_shd;
	};
}
