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
			const auto& mtrls = material::getMaterials();

			// TODO
			// Find material.
			auto found = std::find_if(
				mtrls.begin(), mtrls.end(),
				[&](const material* mtrl)->bool
			{
				if (mtrl->nameString() == mtrlDesc.name) {
					return true;
				}

				return false;
			});

			if (found != mtrls.end()) {
				const auto mtrl = *found;
				const auto& mtrlParam = mtrl->param();

				auto albedo = texture::getTexture(mtrlParam.albedoMap);
				if (albedo) {
					albedo->bindAsGLTexture(0, m_shd);
				}
			}
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

	//////////////////////////////////////////////////////////////

	shader DeformableRenderer::s_shd;

	bool DeformableRenderer::init(
		int width, int height,
		const char* pathVS,
		const char* pathFS)
	{
		return s_shd.init(width, height, pathVS, pathFS);
	}

	void DeformableRenderer::render(deformable* mdl)
	{
		CALL_GL_API(::glClearColor(0, 0.5f, 1.0f, 1.0f));
		CALL_GL_API(::glClearDepthf(1.0f));
		CALL_GL_API(::glClearStencil(0));
		CALL_GL_API(::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

		mdl->render(&s_shd);
	}
}
