#pragma once

#include "deformable/MDLFormat.h"
#include "deformable/DeformMesh.h"
#include "deformable/Skeleton.h"
#include "deformable/SkinningVertex.h"

namespace aten
{
	class shader;
	class DeformAnimation;

	/** メッシュデータ.
	 */
	class deformable {
	public:
		deformable() {}
		~deformable() {}

	public:
		bool read(const char* path);

		void release();

		void update(const mat4& mtxL2W);

		void update(
			const mat4& mtxL2W,
			DeformAnimation* anm,
			real time);

		void render(shader* shd);

		void getGeometryData(
			std::vector<SkinningVertex>& vtx,
			std::vector<uint32_t>& idx) const;

		const std::vector<mat4>& getMatrices() const;

		bool isEnabledForGPUSkinning() const
		{
			return m_mesh.getDesc().isGPUSkinning;
		}

		GeomVertexBuffer& getVBForGPUSkinning()
		{
			return m_mesh.getVBForGPUSkinning();
		}

	private:
		DeformMesh m_mesh;

		// TODO
		Skeleton m_skl;
		SkeletonController m_sklController;
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
