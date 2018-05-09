#include "kernel/pathtracing.h"
#include "kernel/compaction.h"
#include "kernel/pt_common.h"

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

namespace idaten {
	void PathTracing::update(
		GLuint gltex,
		int width, int height,
		const aten::CameraParameter& camera,
		const std::vector<aten::GeomParameter>& shapes,
		const std::vector<aten::MaterialParameter>& mtrls,
		const std::vector<aten::LightParameter>& lights,
		const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
		const std::vector<aten::PrimitiveParamter>& prims,
		uint32_t advancePrimNum,
		const std::vector<aten::vertex>& vtxs,
		uint32_t advanceVtxNum,
		const std::vector<aten::mat4>& mtxs,
		const std::vector<TextureResource>& texs,
		const EnvmapResource& envmapRsc)
	{
		idaten::Renderer::update(
			gltex,
			width, height,
			camera,
			shapes,
			mtrls,
			lights,
			nodes,
			prims, advancePrimNum,
			vtxs, advanceVtxNum,
			mtxs,
			texs, envmapRsc);

		m_sobolMatrices.init(AT_COUNTOF(sobol::Matrices::matrices));
		m_sobolMatrices.writeByNum(sobol::Matrices::matrices, m_sobolMatrices.num());

		auto& r = aten::getRandom();

		m_random.init(width * height);
		m_random.writeByNum(&r[0], width * height);

		m_paths.init(width * height);
	}

	void PathTracing::updateMaterial(const std::vector<aten::MaterialParameter>& mtrls)
	{
		AT_ASSERT(mtrls.size() <= m_mtrlparam.num());

		if (mtrls.size() <= m_mtrlparam.num()) {
			m_mtrlparam.writeByNum(&mtrls[0], (uint32_t)mtrls.size());

			reset();
		}
	}

	void PathTracing::enableRenderAOV(
		GLuint gltexPosition,
		GLuint gltexNormal,
		GLuint gltexAlbedo,
		const aten::vec3& posRange)
	{
		AT_ASSERT(gltexPosition > 0);
		AT_ASSERT(gltexNormal > 0);

		if (!m_enableAOV) {
			m_enableAOV = true;

			m_posRange = posRange;

			m_aovs.resize(3);
			m_aovs[0].init(gltexPosition, CudaGLRscRegisterType::WriteOnly);
			m_aovs[1].init(gltexNormal, CudaGLRscRegisterType::WriteOnly);
			m_aovs[2].init(gltexAlbedo, CudaGLRscRegisterType::WriteOnly);

			m_aovCudaRsc.init(3);
		}
	}

#ifdef __AT_DEBUG__
	static bool doneSetStackSize = false;
#endif

	void PathTracing::render(
		const TileDomain& tileDomain,
		int maxSamples,
		int maxBounce)
	{
#ifdef __AT_DEBUG__
		if (!doneSetStackSize) {
			size_t val = 0;
			cudaThreadGetLimit(&val, cudaLimitStackSize);
			cudaThreadSetLimit(cudaLimitStackSize, val * 4);
			doneSetStackSize = true;
		}
#endif

		m_tileDomain = tileDomain;

		int bounce = 0;

		int width = tileDomain.w;
		int height = tileDomain.h;

		m_isects.init(width * height);
		m_rays.init(width * height);

		m_hitbools.init(width * height);
		m_hitidx.init(width * height);

		m_shadowRays.init(width * height);

		checkCudaErrors(cudaMemset(m_paths.ptr(), 0, m_paths.bytes()));

		m_glimg.map();
		auto outputSurf = m_glimg.bind();

		auto vtxTexPos = m_vtxparamsPos.bind();
		auto vtxTexNml = m_vtxparamsNml.bind();

		{
			std::vector<cudaTextureObject_t> tmp;
			for (int i = 0; i < m_nodeparam.size(); i++) {
				auto nodeTex = m_nodeparam[i].bind();
				tmp.push_back(nodeTex);
			}
			m_nodetex.writeByNum(&tmp[0], (uint32_t)tmp.size());
		}

		if (!m_texRsc.empty())
		{
			std::vector<cudaTextureObject_t> tmp;
			for (int i = 0; i < m_texRsc.size(); i++) {
				auto cudaTex = m_texRsc[i].bind();
				tmp.push_back(cudaTex);
			}
			m_tex.writeByNum(&tmp[0], (uint32_t)tmp.size());
		}

		if (m_enableAOV) {
			std::vector<cudaSurfaceObject_t> tmp;
			for (int i = 0; i < m_aovs.size(); i++) {
				m_aovs[i].map();
				tmp.push_back(m_aovs[i].bind());
			}
			m_aovCudaRsc.writeByNum(&tmp[0], (uint32_t)tmp.size());
		}

		static const int rrBounce = 3;

		auto time = AT_NAME::timer::getSystemTime();

		for (int i = 0; i < maxSamples; i++) {
			onGenPath(
				width, height,
				i, maxSamples,
				vtxTexPos,
				vtxTexNml);

			bounce = 0;

			while (bounce < maxBounce) {
				onHitTest(
					width, height,
					vtxTexPos);
				
				onShadeMiss(width, height, bounce);

				m_compaction.compact(
					m_hitidx,
					m_hitbools,
					nullptr);

				onShade(
					width, height,
					bounce, rrBounce,
					vtxTexPos, vtxTexNml);

				bounce++;
			}
		}

		onGather(outputSurf, m_paths, width, height);

		checkCudaErrors(cudaDeviceSynchronize());

		m_frame++;

		{
			m_vtxparamsPos.unbind();
			m_vtxparamsNml.unbind();

			for (int i = 0; i < m_nodeparam.size(); i++) {
				m_nodeparam[i].unbind();
			}

			for (int i = 0; i < m_texRsc.size(); i++) {
				m_texRsc[i].unbind();
			}

			for (int i = 0; i < m_aovs.size(); i++) {
				m_aovs[i].unbind();
				m_aovs[i].unmap();
			}
		}

		m_glimg.unbind();
		m_glimg.unmap();
	}
}
