#pragma

#include "defs.h"
#include "deformable/MSHFormat.h"
#include "visualizer/GeomDataBuffer.h"

#include <vector>

namespace aten
{
	class FileInputStream;

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
