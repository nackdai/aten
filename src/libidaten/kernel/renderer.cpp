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
		const std::vector<aten::vertex>& vtxs)
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

		for (int i = 0; i < nodes.size(); i++) {
			nodeparam.push_back(idaten::CudaTextureResource());
			nodeparam[i].init((aten::vec4*)&nodes[i][0], sizeof(aten::BVHNode) / sizeof(float4), nodes[i].size());
		}
		nodetex.init(nodes.size());

		vtxparams.init((aten::vec4*)&vtxs[0], sizeof(aten::vertex) / sizeof(float4), vtxs.size());
	}
}
