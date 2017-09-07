#include "filter/taa.h"
#include "visualizer/atengl.h"
#include "math/mat4.h"
#include "texture/texture.h"

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

		return true;
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

		auto enableTAA = m_body->isEnableTAA();
		auto hEnableTAA = this->getHandle("enableTAA");
		CALL_GL_API(::glUniform1i(hEnableTAA, enableTAA));

		auto canShowDiff = m_body->canShowTAADiff();
		auto hShowDiff = this->getHandle("showDiff");
		CALL_GL_API(::glUniform1i(hShowDiff, canShowDiff));
	}

	void TAA::FinalPass::prepareRender(
		const void* pixels,
		bool revert)
	{
		shader::prepareRender(pixels, revert);

		auto prevPass = this->getPrevPass();

		auto tex = prevPass->getFbo().getTexHandle(2);

		texture::bindAsGLTexture(tex, 0, this);

		m_body->m_idx = 1 - m_body->m_idx;
	}
}