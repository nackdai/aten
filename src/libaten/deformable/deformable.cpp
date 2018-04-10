#include "deformable/deformable.h"
#include "deformable/DeformAnimation.h"
#include "misc/stream.h"
#include "visualizer/shader.h"
#include "visualizer/atengl.h"
#include "texture/texture.h"
#include "camera/camera.h"

#include <algorithm>
#include <iterator>

namespace aten
{
	class DeformMeshReadHelper : public IDeformMeshReadHelper {
	public:
		DeformMeshReadHelper() {}
		virtual ~DeformMeshReadHelper() {}

	public:
		virtual void createVAO(
			GeomVertexBuffer* vb,
			const VertexAttrib* attribs,
			uint32_t attribNum) override final
		{
			AT_ASSERT(m_shd);
			vb->createVAOByAttribName(m_shd, attribs, attribNum);
		}

		shader* m_shd{ nullptr };
	};

	bool deformable::read(const char* path)
	{
		// TODO...
		DeformMeshReadHelper helper;
		DeformableRenderer::initDeformMeshReadHelper(&helper);

		FileInputStream file;
		AT_VRETURN_FALSE(file.open(path, "rb"));

		FileInputStream* stream = &file;

		MdlHeader header;
		AT_VRETURN_FALSE(AT_STREAM_READ(stream, &header, sizeof(header)));

		// Mesh.
		{
			MdlChunkHeader meshChunkHeader;
			AT_VRETURN_FALSE(AT_STREAM_READ(stream, &meshChunkHeader, sizeof(meshChunkHeader)));

			if (meshChunkHeader.magicChunk == MdlChunkMagic::Mesh) {
				AT_VRETURN_FALSE(m_mesh.read(stream, &helper));
			}
			else {
				AT_VRETURN_FALSE(false);
			}
		}

		auto pos = stream->curPos();

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

			m_sklController.init(&m_skl);
		}

		return true;
	}

	void deformable::release()
	{
		m_mesh.release();
		m_skl.release();
		m_sklController.release();
	}

	class DeformMeshRenderHelper : public IDeformMeshRenderHelper {
	public:
		DeformMeshRenderHelper(shader* s) : m_shd(s) {}
		virtual ~DeformMeshRenderHelper() {}

		virtual void applyMatrix(uint32_t idx, const mat4& mtx) override final
		{
			if (m_handleMtxJoint < 0) {
				m_handleMtxJoint = m_shd->getHandle("mtxJoint");
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
		if (isEnabledForGPUSkinning()) {
			AT_ASSERT_LOG(false, "Not support for gpu skinning format data.");
			return;
		}

		AT_ASSERT(shd);

		DeformMeshRenderHelper helper(shd);

		m_mesh.render(m_sklController, &helper);
	}

	void deformable::update(const mat4& mtxL2W)
	{
		m_sklController.buildPose(mtxL2W);
	}

	void deformable::update(
		const mat4& mtxL2W,
		DeformAnimation* anm,
		real time)
	{
		anm->applyAnimation(&m_sklController, time);
		m_sklController.buildPose(mtxL2W);
	}

	void deformable::getGeometryData(
		std::vector<SkinningVertex>& vtx,
		std::vector<uint32_t>& idx) const
	{
		m_mesh.getGeometryData(vtx, idx);
	}

	const std::vector<mat4>& deformable::getMatrices() const
	{
		return m_sklController.getMatrices();
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

	void DeformableRenderer::render(
		const camera* cam,
		deformable* mdl)
	{
		CALL_GL_API(::glClearColor(0, 0.5f, 1.0f, 1.0f));
		CALL_GL_API(::glClearDepthf(1.0f));
		CALL_GL_API(::glClearStencil(0));
		CALL_GL_API(::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

		CALL_GL_API(::glEnable(GL_DEPTH_TEST));

		// For Alpha Blend.
		CALL_GL_API(::glEnable(GL_BLEND));
		CALL_GL_API(::glBlendEquation(GL_FUNC_ADD));
		CALL_GL_API(::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

		s_shd.prepareRender(nullptr, false);

		{
			auto camparam = cam->param();

			// TODO
			camparam.znear = real(0.1);
			camparam.zfar = real(10000.0);

			mat4 mtxW2V;
			mat4 mtxV2C;

			mtxW2V.lookat(
				camparam.origin,
				camparam.center,
				camparam.up);

			mtxV2C.perspective(
				camparam.znear,
				camparam.zfar,
				camparam.vfov,
				camparam.aspect);

			aten::mat4 mtxW2C = mtxV2C * mtxW2V;

			auto hMtxW2C = s_shd.getHandle("mtxW2C");
			CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, &mtxW2C.a[0]));

			// TODO
			aten::mat4 mtxL2W;
			auto hMtxL2W = s_shd.getHandle("mtxL2W");
			CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, &mtxL2W.a[0]));
		}

		mdl->render(&s_shd);
	}

	void DeformableRenderer::initDeformMeshReadHelper(DeformMeshReadHelper* helper)
	{
		AT_ASSERT(s_shd.isValid());
		helper->m_shd = &s_shd;
	}
}
