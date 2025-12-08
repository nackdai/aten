#include "ao/ao.h"
#include "kernel/StreamCompaction.h"
#include "kernel/pt_common.h"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"
#include "aten4idaten.h"

namespace idaten {
#ifdef __AT_DEBUG__
    static bool doneSetStackSize = false;
#endif

    void AORenderer::render(
        int32_t width, int32_t height,
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

        m_compaction.init(width * height, 1024);

        m_isects.resize(width * height);
        m_rays.resize(width * height);

        m_hitbools.resize(width * height);
        m_hitidx.resize(width * height);

        m_glimg.map();
        auto outputSurf = m_glimg.bind();

        InitPath(width, height);
        clearPath();

        static const int32_t rrBounce = 3;

        auto seed = AT_NAME::timer::GetCurrMilliseconds();

        for (int32_t i = 0; i < maxSamples; i++) {
            generatePath(
                width, height,
                false,
                i, maxBounce,
                seed);

            bounce = 0;

            while (bounce < maxBounce) {
                hitTest(
                    width, height,
                    bounce);

                missShade(width, height, bounce);

                m_compaction.compact(
                    m_hitidx,
                    m_hitbools,
                    nullptr);

                ShadeAO(
                    width, height,
                    bounce, rrBounce);

                bounce++;
            }
        }

        onGather(outputSurf, width, height, maxSamples);

        checkCudaErrors(cudaDeviceSynchronize());

        m_frame++;

        m_glimg.unbind();
        m_glimg.unmap();
    }
}
