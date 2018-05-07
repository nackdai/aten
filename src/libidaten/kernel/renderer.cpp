#include "kernel/renderer.h"

namespace idaten {
	void Renderer::update(
		GLuint gltex,
		int width, int height,
		const aten::CameraParameter& camera,
		const std::vector<aten::GeomParameter>& shapes,
		const std::vector<aten::MaterialParameter>& mtrls,
		const std::vector<aten::LightParameter>& lights,
		const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
		const std::vector<aten::PrimitiveParamter>& prims,
		const std::vector<aten::vertex>& vtxs,
		uint32_t advanceVtxNum,
		const std::vector<aten::mat4>& mtxs,
		const std::vector<TextureResource>& texs,
		const EnvmapResource& envmapRsc)
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
		m_glimg.init(gltex, CudaGLRscRegisterType::ReadWrite);
#endif

		m_cam.init(sizeof(camera));
		m_cam.writeByNum(&camera, 1);
		m_camParam = camera;

		m_shapeparam.init(shapes.size());
		m_shapeparam.writeByNum(&shapes[0], shapes.size());

		m_mtrlparam.init(mtrls.size());
		m_mtrlparam.writeByNum(&mtrls[0], mtrls.size());

		AT_ASSERT_LOG(!lights.empty(), "No Lights!!");

		m_lightparam.init(lights.size());
		m_lightparam.writeByNum(&lights[0], lights.size());

		if (!prims.empty()) {
			m_primparams.init(prims.size());
			m_primparams.writeByNum(&prims[0], prims.size());
		}

		if (!mtxs.empty()) {
			m_mtxparams.init(mtxs.size());
			m_mtxparams.writeByNum(&mtxs[0], mtxs.size());
		}

		m_nodeparam.resize(nodes.size());
		for (int i = 0; i < nodes.size(); i++) {
			if (!nodes[i].empty()) {
				m_nodeparam[i].init(
					(aten::vec4*)&nodes[i][0], 
					sizeof(aten::GPUBvhNode) / sizeof(float4), 
					nodes[i].size());
			}
		}
		m_nodetex.init(nodes.size());

		if (!vtxs.empty()) {
			// TODO
			std::vector<aten::vec4> pos;
			std::vector<aten::vec4> nml;

			for (const auto& v : vtxs) {
				pos.push_back(aten::vec4(v.pos.x, v.pos.y, v.pos.z, v.uv.x));
				nml.push_back(aten::vec4(v.nml.x, v.nml.y, v.nml.z, v.uv.y));
			}

			m_vtxparamsPos.init((aten::vec4*)&pos[0], 1, pos.size() + advanceVtxNum);
			m_vtxparamsNml.init((aten::vec4*)&nml[0], 1, nml.size() + advanceVtxNum);
		}

		if (!texs.empty()) {
			for (int i = 0; i < texs.size(); i++) {
				m_texRsc.push_back(idaten::CudaTexture());

				// TODO
#if 0
				if (envmapRsc.idx == i) {
					m_texRsc[i].initAsMipmap(texs[i].ptr, texs[i].width, texs[i].height, 100);
				}
				else {
					m_texRsc[i].init(texs[i].ptr, texs[i].width, texs[i].height);
				}
#else
				m_texRsc[i].initAsMipmap(texs[i].ptr, texs[i].width, texs[i].height, 100);
#endif
			}
			m_tex.init(texs.size());

			AT_ASSERT(envmapRsc.idx < texs.size());
			if (envmapRsc.idx < texs.size()) {
				m_envmapRsc = envmapRsc;
			}
		}
	}

	void Renderer::updateCamera(const aten::CameraParameter& camera)
	{
		m_cam.reset();
		m_cam.writeByNum(&camera, 1);

		m_camParam = camera;
	}
}
