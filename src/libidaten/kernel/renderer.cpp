#include "kernel/renderer.h"

namespace idaten {
	void Renderer::update(
		GLuint gltex,
		int width, int height,
		const aten::CameraParameter& camera,
		const std::vector<aten::ShapeParameter>& shapes,
		const std::vector<aten::MaterialParameter>& mtrls,
		const std::vector<aten::LightParameter>& lights,
		const std::vector<std::vector<aten::BVHNode>>& nodes,
		const std::vector<aten::PrimitiveParamter>& prims,
		const std::vector<aten::vertex>& vtxs,
		const std::vector<aten::mat4>& mtxs,
		const std::vector<TextureResource>& texs,
		int envmapIdx)
	{
#if 0
		size_t size_stack = 0;
		checkCudaErrors(cudaThreadGetLimit(&size_stack, cudaLimitStackSize));
		checkCudaErrors(cudaThreadSetLimit(cudaLimitStackSize, 12928));
		checkCudaErrors(cudaThreadGetLimit(&size_stack, cudaLimitStackSize));

		AT_PRINTF("Stack size %d\n", size_stack);
#endif

#if 0
		dst.init(sizeof(float4) * width * height);
#else
		//glimg.init(gltex, CudaGLRscRegisterType::WriteOnly);
		glimg.init(gltex, CudaGLRscRegisterType::ReadWrite);
#endif

		cam.init(sizeof(camera));
		cam.writeByNum(&camera, 1);

		shapeparam.init(shapes.size());
		shapeparam.writeByNum(&shapes[0], shapes.size());

		mtrlparam.init(mtrls.size());
		mtrlparam.writeByNum(&mtrls[0], mtrls.size());

		lightparam.init(lights.size());
		lightparam.writeByNum(&lights[0], lights.size());

		if (!prims.empty()) {
			primparams.init(prims.size());
			primparams.writeByNum(&prims[0], prims.size());
		}

		if (!mtxs.empty()) {
			mtxparams.init(mtxs.size());
			mtxparams.writeByNum(&mtxs[0], mtxs.size());
		}

		for (int i = 0; i < nodes.size(); i++) {
			nodeparam.push_back(idaten::CudaTextureResource());
			nodeparam[i].init((aten::vec4*)&nodes[i][0], sizeof(aten::BVHNode) / sizeof(float4), nodes[i].size());
		}
		nodetex.init(nodes.size());

		if (!vtxs.empty()) {
			// TODO
			std::vector<aten::vec4> pos;
			std::vector<aten::vec4> nml;

			for (const auto& v : vtxs) {
				pos.push_back(aten::vec4(v.pos.x, v.pos.y, v.pos.z, v.uv.x));
				nml.push_back(aten::vec4(v.nml.x, v.nml.y, v.nml.z, v.uv.y));
			}

			vtxparamsPos.init((aten::vec4*)&pos[0], 1, pos.size());
			vtxparamsNml.init((aten::vec4*)&nml[0], 1, nml.size());
		}

		if (!texs.empty()) {
			for (int i = 0; i < texs.size(); i++) {
				texRsc.push_back(idaten::CudaTexture());
				texRsc[i].init(texs[i].ptr, texs[i].width, texs[i].height);
			}
			tex.init(texs.size());

			AT_ASSERT(envmapIdx < texs.size());
			if (envmapIdx < texs.size()) {
				m_envmapIdx = envmapIdx;
			}
		}
	}

	void Renderer::updateCamera(const aten::CameraParameter& camera)
	{
		cam.reset();
		cam.writeByNum(&camera, 1);
	}
}
