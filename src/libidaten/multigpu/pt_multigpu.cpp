#include "multigpu/pt_multigpu.h"

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

namespace idaten
{
#ifdef __AT_DEBUG__
	static bool doneSetStackSize = false;
#endif

	void PathTracingMultiGPU::render(
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

		m_frame++;

		{
			m_vtxparamsPos.unbind();
			m_vtxparamsNml.unbind();

			for (int i = 0; i < m_nodeparam.size(); i++) {
				m_nodeparam[i].unbind();
			}
			m_nodetex.reset();

			for (int i = 0; i < m_texRsc.size(); i++) {
				m_texRsc[i].unbind();
			}
			m_tex.reset();

			for (int i = 0; i < m_aovs.size(); i++) {
				m_aovs[i].unbind();
				m_aovs[i].unmap();
			}
			m_aovCudaRsc.reset();
		}
	}

	void PathTracingMultiGPU::postRender(int width, int height)
	{
		m_glimg.map();
		auto outputSurf = m_glimg.bind();

		width = width > 0 ? width : m_tileDomain.w;
		height = height > 0 ? height : m_tileDomain.h;

		onGather(outputSurf, m_paths, width, height);

		checkCudaErrors(cudaDeviceSynchronize());

		m_glimg.unbind();
		m_glimg.unmap();
	}

	void PathTracingMultiGPU::copy(
		PathTracingMultiGPU& from,
		cudaStream_t stream)
	{
		if (this == &from) {
			AT_ASSERT(false);
			return;
		}

		const auto& srcTileDomain = from.m_tileDomain;
		auto src = from.m_paths.ptr();

		const auto& dstTileDomain = this->m_tileDomain;
		auto dst = this->m_paths.ptr();

		AT_ASSERT(srcTileDomain.w == dstTileDomain.w);

		auto stride = this->m_paths.stride();

		auto offset = srcTileDomain.y * dstTileDomain.w + srcTileDomain.x;
		auto bytes = srcTileDomain.w * srcTileDomain.h * stride;

		checkCudaErrors(cudaMemcpyAsync(dst + offset, src, bytes, cudaMemcpyDefault, stream));
	}
}
