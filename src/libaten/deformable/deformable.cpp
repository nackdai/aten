#include "deformable/deformable.h"
#include "misc/stream.h"
#include "visualizer/shader.h"
#include "visualizer/atengl.h"
#include "texture/texture.h"

namespace aten
{
	bool deformable::read(FileInputStream* stream)
	{
		MdlHeader header;
		AT_VRETURN_FALSE(AT_STREAM_READ(stream, &header, sizeof(header)));

		// Mesh.
		{
			MdlChunkHeader meshChunkHeader;
			AT_VRETURN_FALSE(AT_STREAM_READ(stream, &meshChunkHeader, sizeof(meshChunkHeader)));

			if (meshChunkHeader.magicChunk == MdlChunkMagic::Mesh) {
				AT_VRETURN_FALSE(m_mesh.read(stream));
			}
			else {
				AT_VRETURN_FALSE(false);
			}
		}

		// Skeleton.
		{
			MdlChunkHeader sklChunkHeader;
			AT_VRETURN_FALSE(AT_STREAM_READ(stream, &sklChunkHeader, sizeof(sklChunkHeader)));

			if (sklChunkHeader.magicChunk == MdlChunkMagic::Joint) {
				AT_VRETURN_FALSE(m_skl.read(stream));
			}
			else {
				AT_VRETURN_FALSE(false);
			}
		}

		return true;
	}

	class DeformMeshRenderHelper : public IDeformMeshRenderHelper {
	public:
		DeformMeshRenderHelper(shader* s) : m_shd(s) {}
		virtual ~DeformMeshRenderHelper() {}

		virtual void applyMatrix(uint32_t idx, const mat4& mtx) override final
		{
			if (m_handleMtxJoint < 0) {
				m_handleMtxJoint = m_shd->getHandle("mtxJoints");
				m_mtxs.reserve(4);
			}

			m_mtxs.push_back(mtx);
		}

		virtual void applyMaterial(const MeshMaterial& mtrlDesc) override final
		{
			// TODO
			// Find material.
		}

		virtual void commitChanges() override final
		{
			AT_ASSERT(m_handleMtxJoint >= 0);

			uint32_t mtxNum = (uint32_t)m_mtxs.size();

			CALL_GL_API(::glUniformMatrix4fv(m_handleMtxJoint, mtxNum, GL_TRUE, (const GLfloat*)&m_mtxs[0]));

			m_mtxs.clear();
		}

		shader* m_shd{ nullptr };
		int m_handleMtxJoint{ -1 };
		std::vector<mat4> m_mtxs;
	};

	void deformable::render(shader* shd)
	{
		AT_ASSERT(shd);

		shd->prepareRender(nullptr, false);

		DeformMeshRenderHelper helper(shd);

		m_mesh.render(m_skl, &helper);
	}
}
