#include "visualizer/atengl.h"
#include "visualizer/MultiPassPostProc.h"

namespace aten {
	void MultiPassPostProc::prepareRender(
		const void* pixels,
		bool revert)
	{
		for (int i = 0; i < m_passes.size(); i++) {
			auto* pass = m_passes[i];
			visualizer::PostProc* prevPass = nullptr;

			if (i > 0) {
				prevPass = m_passes[i - 1];

				// Set FBO as source texture.
				prevPass->getFbo().bindAsTexture();
			}

			pass->prepareRender(prevPass, pixels, revert);

			if (pass->getFbo().isValid()) {
				// Set FBO.
				pass->getFbo().setFBO();
			}
			else {
				// Set default frame buffer.
				CALL_GL_API(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
			}

			CALL_GL_API(::glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
		}
	}

	bool MultiPassPostProc::addPass(visualizer::PostProc* pass)
	{
		if (m_passes.size() > 0) {
			// Create fbo to connect between post-processes.
			auto idx = m_passes.size() - 1;
			auto* prevPass = m_passes[idx];
			auto outFmt = prevPass->outFormat();

			// Check in-out format.
			auto inFmt = pass->inFormat();
			AT_VRETURN(inFmt == outFmt, false);

			auto& fbo = prevPass->getFbo();

			auto width = prevPass->getOutWidth();
			auto height = prevPass->getOutHeight();

			// Create FBO.
			AT_VRETURN(fbo.init(width, height, outFmt), false);
		}

		m_passes.push_back(pass);

		return true;
	}
}