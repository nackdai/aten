#include "visualizer/RasterizeRenderer.h"
#include "visualizer/atengl.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "math/mat4.h"
#include "sampler/cmj.h"

namespace aten {
	shader ResterizeRenderer::s_shader;

	static int s_width = 0;
	static int s_height = 0;

	bool ResterizeRenderer::init(
		int width, int height,
		const char* pathVS,
		const char* pathFS)
	{
		s_width = width;
		s_height = height;

		return s_shader.init(width, height, pathVS, pathFS);
	}

	bool ResterizeRenderer::init(
		int width, int height,
		const char* pathVS,
		const char* pathGS,
		const char* pathFS)
	{
		s_width = width;
		s_height = height;

		return s_shader.init(width, height, pathVS, pathGS, pathFS);
	}

	void ResterizeRenderer::draw(
		int frame,
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
		// For TAA.
		{
			CMJ sampler;
			auto rnd = getRandom(0);
			auto scramble = rnd * 0x1fe3434f * ((frame + 331 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
			sampler.init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 4 + 300, scramble);

			auto smpl = sampler.nextSample2D();

			aten::mat4 mtxOffset;
			mtxOffset.asTrans(aten::vec3(
				smpl.x / s_width, 
				smpl.y / s_height, 
				real(0)));

			auto hMtxOffset = s_shader.getHandle("mtxOffset");
			CALL_GL_API(::glUniformMatrix4fv(hMtxOffset, 1, GL_TRUE, &mtxOffset.a[0]));
		}

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

		scene->draw([&](const aten::mat4& mtxL2W, int objid, int primid) {
			auto hMtxL2W = s_shader.getHandle("mtxL2W");
			CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, &mtxL2W.a[0]));

			auto hObjId = s_shader.getHandle("objid");
			CALL_GL_API(::glUniform1i(hObjId, objid));

			auto hPrimId = s_shader.getHandle("primid");
			CALL_GL_API(::glUniform1i(hPrimId, primid));
		});

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
