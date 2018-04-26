#include "multigpu/svgf_multigpu.h"

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

namespace idaten
{
	static bool doneSetStackSize = false;

	void SVGFPathTracingMultiGPU::render(
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

		int bounce = 0;

		int width = tileDomain.w;
		int height = tileDomain.h;

		m_isects.init(width * height);
		m_rays.init(width * height);

		m_shadowRays.init(width * height * ShadowRayNum);

		onInit(width, height);

		CudaGLResourceMapper rscmap(&m_glimg);
		auto outputSurf = m_glimg.bind();

		auto vtxTexPos = m_vtxparamsPos.bind();
		auto vtxTexNml = m_vtxparamsNml.bind();

		// TODO
		// Textureメモリのバインドによる取得されるcudaTextureObject_tは変化しないので,値を一度保持しておけばいい.
		// 現時点では最初に設定されたものが変化しない前提でいるが、入れ替えなどの変更があった場合はこの限りではないので、何かしらの対応が必要.

		if (!m_isListedTextureObject)
		{
			{
				std::vector<cudaTextureObject_t> tmp;
				for (int i = 0; i < m_nodeparam.size(); i++) {
					auto nodeTex = m_nodeparam[i].bind();
					tmp.push_back(nodeTex);
				}
				m_nodetex.writeByNum(&tmp[0], tmp.size());
			}

			if (!m_texRsc.empty())
			{
				std::vector<cudaTextureObject_t> tmp;
				for (int i = 0; i < m_texRsc.size(); i++) {
					auto cudaTex = m_texRsc[i].bind();
					tmp.push_back(cudaTex);
				}
				m_tex.writeByNum(&tmp[0], tmp.size());
			}

			m_isListedTextureObject = true;
		}
		else {
			for (int i = 0; i < m_nodeparam.size(); i++) {
				auto nodeTex = m_nodeparam[i].bind();
			}
			for (int i = 0; i < m_texRsc.size(); i++) {
				auto cudaTex = m_texRsc[i].bind();
			}
		}

		m_hitbools.init(width * height);
		m_hitidx.init(width * height);

		m_compaction.init(
			width * height,
			1024);

		onClear();

		onRender(
			tileDomain,
			width, height, maxSamples, maxBounce,
			outputSurf,
			vtxTexPos,
			vtxTexNml);

		{
			m_mtxPrevW2V = m_mtxW2V;

			//checkCudaErrors(cudaDeviceSynchronize());

			// Toggle aov buffer pos.
			m_curAOVPos = 1 - m_curAOVPos;

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
			}
		}

		// Keep specified tile domain.
		m_tileDomain = tileDomain;
	}

	void SVGFPathTracingMultiGPU::onRender(
		const TileDomain& tileDomain,
		int width, int height,
		int maxSamples,
		int maxBounce,
		cudaSurfaceObject_t outputSurf,
		cudaTextureObject_t vtxTexPos,
		cudaTextureObject_t vtxTexNml)
	{
		static const int rrBounce = 3;

		// Set bounce count to 1 forcibly, aov render mode.
		maxBounce = (m_mode == Mode::AOVar ? 1 : maxBounce);

		auto time = AT_NAME::timer::getSystemTime();

		for (int i = 0; i < maxSamples; i++) {
			int seed = time.milliSeconds;
			//int seed = 0;

			m_tileDomain = tileDomain;

			onGenPath(
				i, maxSamples,
				seed,
				vtxTexPos,
				vtxTexNml);

			int bounce = 0;

			// NOTE
			// ここから先ではオフセットさせない.
			m_tileDomain.x = 0;
			m_tileDomain.y = 0;

			while (bounce < maxBounce) {
				onHitTest(
					width, height,
					bounce,
					vtxTexPos);

				onShadeMiss(width, height, bounce);

				int hitcount = 0;
				m_compaction.compact(
					m_hitidx,
					m_hitbools);

				//AT_PRINTF("%d\n", hitcount);

				onShade(
					outputSurf,
					width, height,
					bounce, rrBounce,
					vtxTexPos, vtxTexNml);

				bounce++;
			}
		}

		if (m_mode == Mode::PT) {
			onGather(outputSurf, width, height, maxSamples);
		}
		else if (m_mode == Mode::AOVar) {
			onDisplayAOV(outputSurf, width, height, vtxTexPos);
		}
		else {
			if (isFirstFrame()) {
				onGather(outputSurf, width, height, maxSamples);
			}
			else {
				onCopyBufferForTile(width, height);
			}
		}
	}

	void SVGFPathTracingMultiGPU::postRender(int width, int height)
	{
		m_glimg.map();
		auto outputSurf = m_glimg.bind();

		// NOTE
		// renderで切り替えられているが、本来はdenoise後に切り替えるので、ここで一度元に戻す.
		m_curAOVPos = 1 - m_curAOVPos;

		onDenoise(
			TileDomain(0, 0, width, height),
			width, height,
			outputSurf);

		if (m_mode == Mode::SVGF)
		{
			onAtrousFilter(outputSurf, width, height);
			onCopyFromTmpBufferToAov(width, height);
		}

		m_glimg.unbind();
		m_glimg.unmap();

		// NOTE
		// 一度元に戻されたものを切り替えられた状態に戻す.
		m_curAOVPos = 1 - m_curAOVPos;
	}

	void SVGFPathTracingMultiGPU::copyFrom(SVGFPathTracingMultiGPU& tracer)
	{
		if (this == &tracer) {
			AT_ASSERT(false);
			return;
		}

		const auto& srcTileDomain = tracer.m_tileDomain;
		

		const auto& dstTileDomain = this->m_tileDomain;

		AT_ASSERT(srcTileDomain.w == dstTileDomain.w);

		auto offset = srcTileDomain.y * dstTileDomain.w + srcTileDomain.x;

		// NOTE
		// すでに切り替えられているが、切り替え前のものを参照したいので、元に戻す.
		auto cur = 1 - m_curAOVPos;
		
		// Notmal & Depth.
		{
			auto src = tracer.m_aovNormalDepth[cur].ptr();
			auto dst = this->m_aovNormalDepth[cur].ptr();
		
			auto stride = this->m_aovNormalDepth[cur].stride();
			auto bytes = srcTileDomain.w * srcTileDomain.h * stride;
			
			checkCudaErrors(cudaMemcpyAsync(dst + offset, src, bytes, cudaMemcpyDefault));
		}

		// Texture color & Temporal weight.
		{
			auto src = tracer.m_aovTexclrTemporalWeight[cur].ptr();
			auto dst = this->m_aovTexclrTemporalWeight[cur].ptr();

			auto stride = this->m_aovTexclrTemporalWeight[cur].stride();
			auto bytes = srcTileDomain.w * srcTileDomain.h * stride;

			checkCudaErrors(cudaMemcpyAsync(dst + offset, src, bytes, cudaMemcpyDefault));
		}
		
		// Color & Variance.
		{
			auto src = tracer.m_aovColorVariance[cur].ptr();
			auto dst = this->m_aovColorVariance[cur].ptr();

			auto stride = this->m_aovColorVariance[cur].stride();
			auto bytes = srcTileDomain.w * srcTileDomain.h * stride;

			checkCudaErrors(cudaMemcpyAsync(dst + offset, src, bytes, cudaMemcpyDefault));
		}

		// Moment & Mesh id.
		{
			auto src = tracer.m_aovMomentMeshid[cur].ptr();
			auto dst = this->m_aovMomentMeshid[cur].ptr();

			auto stride = this->m_aovMomentMeshid[cur].stride();
			auto bytes = srcTileDomain.w * srcTileDomain.h * stride;

			checkCudaErrors(cudaMemcpyAsync(dst + offset, src, bytes, cudaMemcpyDefault));
		}
	}
}
