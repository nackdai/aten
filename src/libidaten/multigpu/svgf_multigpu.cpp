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

	}
}
