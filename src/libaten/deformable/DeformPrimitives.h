#pragma

#include "defs.h"
#include "deformable/MSHFormat.h"
#include "visualizer/GeomDataBuffer.h"
#include "math/mat4.h"
#include "material/material.h"

#include <vector>
#include <functional>

namespace aten
{
	class FileInputStream;
	class Skeleton;

	class IDeformMeshRenderHelper {
	protected:
		IDeformMeshRenderHelper() {}
		virtual ~IDeformMeshRenderHelper() {}

	public:
		virtual void applyMatrix(uint32_t idx, const mat4& mtx) = 0;
		virtual void applyMaterial(const MeshMaterial& mtrlDesc) = 0;
		virtual void commitChanges() = 0;
	};

	/** プリミティブデータ.
	 *
	 * メッシュデータの最小単位.
	 */
	class DeformPrimitives {
		friend class DeformMeshSet;

	public:
		DeformPrimitives() {}
		~DeformPrimitives() {}

	private:
		bool read(FileInputStream* stream);

		void render(
			const Skeleton& skeleton,
			IDeformMeshRenderHelper* helper);

		const PrimitiveSet& getDesc() const
		{
			return m_desc;
		}

		void setVB(GeomVertexBuffer* vb)
		{
			m_vb = vb;
		}

	private:
		PrimitiveSet m_desc;

		// ジョイントインデックス.
		std::vector<int16_t> m_joints;

		GeomVertexBuffer* m_vb{ nullptr };
		GeomIndexBuffer m_ib;
	};
}
