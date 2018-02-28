#pragma once

#include "deformable/DeformPrimitives.h"

namespace aten
{
	class FileInputStream;
	class SkeletonController;

	class IDeformMeshReadHelper {
	protected:
		IDeformMeshReadHelper() {}
		virtual ~IDeformMeshReadHelper() {}

	public:
		virtual void createVAO(
			GeomVertexBuffer* vb,
			const VertexAttrib* attribs, 
			uint32_t attribNum) = 0;
	};

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
			IDeformMeshReadHelper* helper,
			std::vector<GeomVertexBuffer>& vbs);

		void render(
			const SkeletonController& skeleton,
			IDeformMeshRenderHelper* helper);

	private:
		MeshSet m_desc;

		std::vector<DeformPrimitives> m_prims;
	};
}
