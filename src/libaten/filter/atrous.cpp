#include "filter/atrous.h"

#include "visualizer/atengl.h"

namespace aten {
	bool ATrousDenoiser::init(
		int width, int height,
		const char* vsPath, const char* fsPath,
		const char* finalVsPath, const char* finalFsPath)
	{
		m_normal.init(width, height, 4);
		m_pos.init(width, height, 4);
		m_albedo.init(width, height, 4);

		m_normal.initAsGLTexture();
		m_pos.initAsGLTexture();
		m_albedo.initAsGLTexture();

		for (int i = 0; i < ITER; i++) {
			auto res = m_pass[i].init(
				width, height,
				vsPath, fsPath);
			AT_ASSERT(res);

			m_pass[i].m_body = this;
			m_pass[i].m_idx = i;

			m_pass[i].getFbo().asMulti(2);

			addPass(&m_pass[i]);
		}

		{
			auto res = m_final.init(
				width, height,
				finalVsPath, finalFsPath);
			AT_ASSERT(res);

			m_final.m_body = this;

			addPass(&m_final);
		}

		return true;
	}

	void ATrousDenoiser::ATrousPass::prepareRender(
		const void* pixels,
		bool revert)
	{
		shader::prepareRender(pixels, revert);

		// Bind source tex handle.
		if (m_idx == 0)
		{
			GLuint srcTexHandle = visualizer::getTexHandle();
			auto prevPass = m_body->getPrevPass();
			if (prevPass) {
				srcTexHandle = prevPass->getFbo().getTexHandle();
			}

			texture::bindAsGLTexture(srcTexHandle, 0, this);
		}
		else {
			auto prevPass = getPrevPass();
			auto texHandle = prevPass->getFbo().getTexHandle();

			texture::bindAsGLTexture(texHandle, 0, this);
		}

		// Bind G-Buffer.
		m_body->m_normal.bindAsGLTexture(1, this);
		m_body->m_pos.bindAsGLTexture(2, this);

		int stepScale = 1 << m_idx;

		auto hStepScape = this->getHandle("stepScale");
		CALL_GL_API(::glUniform1i(hStepScape, stepScale));

		// TODO
		// Sigma.
	}

	void ATrousDenoiser::ATrousFinalPass::prepareRender(
		const void* pixels,
		bool revert)
	{
		shader::prepareRender(pixels, revert);

		// coarse.
		auto tex = m_body->m_pass[ITER - 1].getFbo().getTexHandle(0);
		texture::bindAsGLTexture(tex, 0, this);

#if 0
		// detail.
		for (int i = 0; i < ITER; i++) {
			auto detailTex = m_body->m_pass[i].getFbo().getTexHandle(1);
			texture::bindAsGLTexture(tex, i + 1, this);
		}
#else
		// albedo.
		m_body->m_albedo.bindAsGLTexture(1, this);
#endif
	}
}