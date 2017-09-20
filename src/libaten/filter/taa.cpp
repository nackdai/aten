#include "filter/taa.h"
#include "visualizer/atengl.h"
#include "math/mat4.h"
#include "texture/texture.h"

#pragma optimize( "", off)

namespace aten
{
	bool TAA::init(
		int width, int height,
		const char* taaVsPath, const char* taaFsPath,
		const char* finalVsPath, const char* finalFsPath)
	{
		m_taa.init(width, height, taaVsPath, taaFsPath);
		m_taa.m_body = this;

		// For MRT.
		m_taa.getFbo().asMulti(3);

		auto func = std::bind(&TAA::prepareFbo, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
		m_taa.getFbo().setPrepareFboFunction(func);

		m_final.init(width, height, finalVsPath, finalFsPath);
		m_final.m_body = this;

		addPass(&m_taa);
		addPass(&m_final);

		m_aovTex.initAsGLTexture(width, height);

		return true;
	}

	void TAA::update(
		uint32_t frame,
		const PinholeCamera& cam)
	{
		if (frame > 1) {
			m_mtxPrevW2V = m_mtxW2V;
		}

		auto camparam = cam.param();

		// TODO
		camparam.znear = real(0.1);
		camparam.zfar = real(10000.0);

		m_mtxW2V.lookat(
			camparam.origin,
			camparam.center,
			camparam.up);

		m_mtxV2C.perspective(
			camparam.znear,
			camparam.zfar,
			camparam.vfov,
			camparam.aspect);

		m_mtxC2V = m_mtxV2C;
		m_mtxC2V.invert();

		m_mtxV2W = m_mtxW2V;
		m_mtxV2W.invert();

		if (frame == 1) {
			m_mtxPrevW2V = m_mtxW2V;
		}
	}

	void TAA::prepareFbo(const uint32_t* tex, int num, std::vector<uint32_t>& comps)
	{
		if (comps.empty()) {
			comps.resize(2);
		}

		int cur = m_idx;
		int next = 1 - cur;

		// Accumulatuin buffer.(for next frame)
		CALL_GL_API(glFramebufferTexture2D(
			GL_FRAMEBUFFER,
			GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_2D,
			tex[next],
			0));

		// Destination.
		CALL_GL_API(glFramebufferTexture2D(
			GL_FRAMEBUFFER,
			GL_COLOR_ATTACHMENT1,
			GL_TEXTURE_2D,
			tex[2],
			0));

		comps[0] = GL_COLOR_ATTACHMENT0;
		comps[1] = GL_COLOR_ATTACHMENT1;
	}

	void TAA::TAAPass::prepareRender(
		const void* pixels,
		bool revert)
	{
		shader::prepareRender(pixels, revert);

		GLuint srcTexHandle = visualizer::getTexHandle();
		texture::bindAsGLTexture(srcTexHandle, 0, this);

		int cur = m_body->m_idx;

		auto texAcc = getFbo().getTexHandle(cur);
		texture::bindAsGLTexture(texAcc, 1, this);

		m_body->m_aovTex.bindAsGLTexture(2, this);

		auto enableTAA = m_body->isEnableTAA();
		auto hEnableTAA = this->getHandle("enableTAA");
		CALL_GL_API(::glUniform1i(hEnableTAA, enableTAA));

		auto canShowDiff = m_body->canShowTAADiff();
		auto hShowDiff = this->getHandle("showDiff");
		CALL_GL_API(::glUniform1i(hShowDiff, canShowDiff));

		auto hMtxC2V = this->getHandle("mtxC2V");
		CALL_GL_API(::glUniformMatrix4fv(hMtxC2V, 1, GL_TRUE, &m_body->m_mtxC2V.a[0]));

		auto hMtxV2C = this->getHandle("mtxV2C");
		CALL_GL_API(::glUniformMatrix4fv(hMtxV2C, 1, GL_TRUE, &m_body->m_mtxV2C.a[0]));

		auto hMtxV2W = this->getHandle("mtxV2W");
		CALL_GL_API(::glUniformMatrix4fv(hMtxV2W, 1, GL_TRUE, &m_body->m_mtxV2W.a[0]));

		auto hPrevMtxW2V = this->getHandle("mtxPrevW2V");
		CALL_GL_API(::glUniformMatrix4fv(hPrevMtxW2V, 1, GL_TRUE, &m_body->m_mtxPrevW2V.a[0]));
	}

	void TAA::FinalPass::prepareRender(
		const void* pixels,
		bool revert)
	{
		shader::prepareRender(pixels, revert);

		auto prevPass = this->getPrevPass();

		auto tex = prevPass->getFbo().getTexHandle(2);

		texture::bindAsGLTexture(tex, 0, this);

		m_body->m_aovTex.bindAsGLTexture(1, this);

		m_body->m_idx = 1 - m_body->m_idx;
	}
}