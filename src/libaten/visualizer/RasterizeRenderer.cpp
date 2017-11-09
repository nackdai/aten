#include "visualizer/RasterizeRenderer.h"
#include "visualizer/atengl.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "math/mat4.h"

namespace aten {
	shader ResterizeRenderer::s_shader;

	bool ResterizeRenderer::init(
		int width, int height,
		const char* pathVS,
		const char* pathFS)
	{
		return s_shader.init(width, height, pathVS, pathFS);
	}

	void ResterizeRenderer::draw(
		scene* scene,
		const camera* cam,
		FBO* fbo/*= nullptr*/)
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

		s_shader.prepareRender(nullptr, false);

		auto hMtxW2C = s_shader.getHandle("mtxW2C");
		CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, &mtxW2C.a[0]));

		// TODO
		// L2W‚ÍinstanceƒNƒ‰ƒX‚Å•ÛŽ‚³‚ê‚Ä‚¢‚é.
		aten::mat4 mtxL2W;
		auto hMtxL2W = s_shader.getHandle("mtxL2W");
		CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, &mtxL2W.a[0]));

		if (fbo) {
			// TODO
			// Œ»Žž“_‚ÌFBO‚ÌŽÀ‘•‚ÍDepth‚ðŽ‚Â‚æ‚¤‚É‚È‚Á‚Ä‚¢‚È‚¢.
			AT_ASSERT(fbo->isValid());
			fbo->setFBO();
		}
		else {
			// Set default frame buffer.
			CALL_GL_API(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
		}

		{
			CALL_GL_API(::glEnable(GL_DEPTH_TEST));
			CALL_GL_API(::glEnable(GL_CULL_FACE));
		}

		// Clear.
		{
			CALL_GL_API(::glClearColor(0.0f, 0.0f, 0.0f, 0.0f));
			CALL_GL_API(::glClearDepthf(1.0f));
			CALL_GL_API(::glClearStencil(0));
			CALL_GL_API(::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
		}
		
		// TODO
		// ‚±‚±‚Å‚â‚é?
		static bool isInitVB = false;
		if (!isInitVB) {
			VertexManager::build();
			isInitVB = true;
		}

		scene->draw();

		if (fbo) {
			// Set default frame buffer.
			CALL_GL_API(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
		}

		{
			CALL_GL_API(::glDisable(GL_DEPTH_TEST));
			CALL_GL_API(::glDisable(GL_CULL_FACE));
		}
	}
}
