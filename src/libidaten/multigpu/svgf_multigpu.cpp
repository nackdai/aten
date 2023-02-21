#include "multigpu/svgf_multigpu.h"

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "kernel/pt_standard_impl.h"

#include "aten4idaten.h"

namespace idaten
{
    static bool doneSetStackSize = false;

    void SVGFPathTracingMultiGPU::render(
        const TileDomain& tileDomain,
        int32_t maxSamples,
        int32_t maxBounce)
    {
#ifdef __AT_DEBUG__
        if (!doneSetStackSize) {
            size_t val = 0;
            cudaThreadGetLimit(&val, cudaLimitStackSize);
            cudaThreadSetLimit(cudaLimitStackSize, val * 4);
            doneSetStackSize = true;
        }
#endif

        int32_t bounce = 0;

        int32_t width = tileDomain.w;
        int32_t height = tileDomain.h;

        m_isects.init(width * height);
        m_rays.init(width * height);

        m_shadowRays.init(width * height * ShadowRayNum);

        initPath(width, height);

        auto vtxTexPos = m_vtxparamsPos.bind();
        auto vtxTexNml = m_vtxparamsNml.bind();

        // Disable SSRT hit test always.
        m_canSSRTHitTest = false;

        // TODO
        // Textureメモリのバインドによる取得されるcudaTextureObject_tは変化しないので,値を一度保持しておけばいい.
        // 現時点では最初に設定されたものが変化しない前提でいるが、入れ替えなどの変更があった場合はこの限りではないので、何かしらの対応が必要.

        if (!m_isListedTextureObject)
        {
            {
                std::vector<cudaTextureObject_t> tmp;
                for (int32_t i = 0; i < m_nodeparam.size(); i++) {
                    auto nodeTex = m_nodeparam[i].bind();
                    tmp.push_back(nodeTex);
                }
                m_nodetex.writeFromHostToDeviceByNum(&tmp[0], tmp.size());
            }

            if (!m_texRsc.empty())
            {
                std::vector<cudaTextureObject_t> tmp;
                for (int32_t i = 0; i < m_texRsc.size(); i++) {
                    auto cudaTex = m_texRsc[i].bind();
                    tmp.push_back(cudaTex);
                }
                m_tex.writeFromHostToDeviceByNum(&tmp[0], tmp.size());
            }

            m_isListedTextureObject = true;
        }
        else {
            for (int32_t i = 0; i < m_nodeparam.size(); i++) {
                auto nodeTex = m_nodeparam[i].bind();
            }
            for (int32_t i = 0; i < m_texRsc.size(); i++) {
                auto cudaTex = m_texRsc[i].bind();
            }
        }

        cudaSurfaceObject_t outputSurf = (cudaSurfaceObject_t)0;
        if (m_mode == Mode::PT) {
            m_glimg.map();
            outputSurf = m_glimg.bind();
        }

        m_hitbools.init(width * height);
        m_hitidx.init(width * height);

        m_compaction.init(
            width * height,
            1024);

        clearPath();

        onRender(
            tileDomain,
            width, height, maxSamples, maxBounce,
            outputSurf,
            vtxTexPos,
            vtxTexNml);

        //checkCudaErrors(cudaDeviceSynchronize());

        {
            m_mtxPrevW2V = m_mtxW2V;

            //checkCudaErrors(cudaDeviceSynchronize());

            // Toggle aov buffer pos.
            m_curAOVPos = 1 - m_curAOVPos;

            m_frame++;

            {
                m_vtxparamsPos.unbind();
                m_vtxparamsNml.unbind();

                for (int32_t i = 0; i < m_nodeparam.size(); i++) {
                    m_nodeparam[i].unbind();
                }

                for (int32_t i = 0; i < m_texRsc.size(); i++) {
                    m_texRsc[i].unbind();
                }
            }
        }

        // Keep specified tile domain.
        m_tileDomain = tileDomain;

        if (m_mode == Mode::PT) {
            m_glimg.unbind();
            m_glimg.unmap();
        }
    }

    void SVGFPathTracingMultiGPU::onRender(
        const TileDomain& tileDomain,
        int32_t width, int32_t height,
        int32_t maxSamples,
        int32_t maxBounce,
        cudaSurfaceObject_t outputSurf,
        cudaTextureObject_t vtxTexPos,
        cudaTextureObject_t vtxTexNml)
    {
        static const int32_t rrBounce = 3;

        // Set bounce count to 1 forcibly, aov render mode.
        maxBounce = (m_mode == Mode::AOVar ? 1 : maxBounce);

        auto time = AT_NAME::timer::getSystemTime();

        for (int32_t i = 0; i < maxSamples; i++) {
            //int32_t seed = time.milliSeconds;
            int32_t seed = 0;

            m_tileDomain = tileDomain;

            generatePath(
                m_mode == Mode::AOVar,
                i, maxBounce,
                seed,
                vtxTexPos,
                vtxTexNml);

            int32_t bounce = 0;

            int32_t offsetX = m_tileDomain.x;
            int32_t offsetY = m_tileDomain.y;

            // NOTE
            // ここから先ではオフセットさせない.
            m_tileDomain.x = 0;
            m_tileDomain.y = 0;

            while (bounce < maxBounce) {
                onHitTest(
                    width, height,
                    bounce,
                    vtxTexPos);

                missShade(
                    width, height, bounce,
                    offsetX, offsetY);

                int32_t hitcount = 0;
                m_compaction.compact(
                    m_hitidx,
                    m_hitbools);

                //AT_PRINTF("%d\n", hitcount);

                onShade(
                    outputSurf,
                    width, height,
                    i,
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

    void SVGFPathTracingMultiGPU::postRender(int32_t width, int32_t height)
    {
        m_glimg.map();
        auto outputSurf = m_glimg.bind();

        // NOTE
        // renderで切り替えられているが、本来はdenoise後に切り替えるので、ここで一度元に戻す.
        auto keepCurAovPos = m_curAOVPos;
        m_curAOVPos = 1 - m_curAOVPos;

        auto keepFrame = m_frame;
        m_frame = (m_frame > 1) ? m_frame - 1 : m_frame;

        auto keepTileDomain = m_tileDomain;
        m_tileDomain = TileDomain(0, 0, width, height);

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

        // Return to kept value.
        m_curAOVPos = keepCurAovPos;
        m_frame = keepFrame;
        m_tileDomain = keepTileDomain;
    }

    void SVGFPathTracingMultiGPU::copy(
        SVGFPathTracingMultiGPU& from,
        cudaStream_t stream)
    {
        if (this == &from) {
            AT_ASSERT(false);
            return;
        }

        const auto& srcTileDomain = from.m_tileDomain;
        const auto& dstTileDomain = this->m_tileDomain;

        AT_ASSERT(srcTileDomain.w == dstTileDomain.w);

        auto offset = srcTileDomain.y * dstTileDomain.w + srcTileDomain.x;

        // NOTE
        // すでに切り替えられているが、切り替え前のものを参照したいので、元に戻す.
        auto cur = 1 - m_curAOVPos;

        // Notmal & Depth.
        {
            auto src = from.aov_[cur].normal_depth().ptr();
            auto dst = this->aov_[cur].normal_depth().ptr();

            auto stride = this->aov_[cur].normal_depth().stride();
            auto bytes = srcTileDomain.w * srcTileDomain.h * stride;

            checkCudaErrors(cudaMemcpyAsync(dst + offset, src, bytes, cudaMemcpyDefault, stream));
        }

        // Texture color & Temporal weight.
        {
            auto src = from.aov_[cur].albedo_meshid().ptr();
            auto dst = this->aov_[cur].albedo_meshid().ptr();

            auto stride = this->aov_[cur].albedo_meshid().stride();
            auto bytes = srcTileDomain.w * srcTileDomain.h * stride;

            checkCudaErrors(cudaMemcpyAsync(dst + offset, src, bytes, cudaMemcpyDefault, stream));
        }

        // Color & Variance.
        if (this->isFirstFrame())
        {
            auto src = from.aov_[cur].get<idaten::SVGFPathTracing::AOVBuffer::ColorVariance>().ptr();
            auto dst = this->aov_[cur].get<idaten::SVGFPathTracing::AOVBuffer::ColorVariance>().ptr();

            auto stride = this->aov_[cur].get<idaten::SVGFPathTracing::AOVBuffer::ColorVariance>().stride();
            auto bytes = srcTileDomain.w * srcTileDomain.h * stride;

            checkCudaErrors(cudaMemcpyAsync(dst + offset, src, bytes, cudaMemcpyDefault, stream));
        }

        // Moment & Mesh id.
        if (this->isFirstFrame())
        {
            auto src = from.aov_[cur].get<idaten::SVGFPathTracing::AOVBuffer::MomentTemporalWeight>().ptr();
            auto dst = this->aov_[cur].get<idaten::SVGFPathTracing::AOVBuffer::MomentTemporalWeight>().ptr();

            auto stride = this->aov_[cur].get<idaten::SVGFPathTracing::AOVBuffer::MomentTemporalWeight>().stride();
            auto bytes = srcTileDomain.w * srcTileDomain.h * stride;

            checkCudaErrors(cudaMemcpyAsync(dst + offset, src, bytes, cudaMemcpyDefault, stream));
        }

        {
            auto src = from.m_tmpBuf.ptr();
            auto dst = this->m_tmpBuf.ptr();

            auto stride = this->m_tmpBuf.stride();
            auto bytes = srcTileDomain.w * srcTileDomain.h * stride;

            checkCudaErrors(cudaMemcpyAsync(dst + offset, src, bytes, cudaMemcpyDefault, stream));
        }
    }
}
