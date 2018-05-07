#pragma once

#include "deformable/DeformPrimitives.h"
#include "geometry/geombase.h"

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
	class DeformMeshSet : public geombase {
		friend class DeformMeshGroup;

	public:
		DeformMeshSet() {}
		~DeformMeshSet() {}

	private:
		bool read(
			FileInputStream* stream,
			IDeformMeshReadHelper* helper,
			std::vector<GeomVertexBuffer>& vbs,
			bool isGPUSkinning);

		void setExternalVertexBuffer(GeomMultiVertexBuffer& vb);

		void render(
			const SkeletonController& skeleton,
			IDeformMeshRenderHelper* helper,
			bool isGPUSkinning);

		const std::vector<DeformPrimitives>& getPrimitives() const
		{
			return m_prims;
		}

		const MeshSet& getDesc() const
		{
			return m_desc;
		}

	private:
		MeshSet m_desc;

		std::vector<DeformPrimitives> m_prims;
	};
}
