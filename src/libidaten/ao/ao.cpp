#include "ao/ao.h"
#include "kernel/StreamCompaction.h"
#include "kernel/pt_common.h"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"
#include "aten4idaten.h"

namespace idaten {
    void AORenderer::OnRender(
        int32_t width, int32_t height,
        int32_t maxSamples,
        int32_t maxBounce,
        cudaSurfaceObject_t outputSurf)
    {
        static const int32_t rrBounce = 3;

        auto seed = AT_NAME::timer::GetCurrMilliseconds();

        for (int32_t i = 0; i < maxSamples; i++) {
            generatePath(
                width, height,
                false,
                i, maxBounce,
                seed);

            hitTest(width, height, 0);

            m_compaction.compact(
                m_hitidx,
                m_hitbools,
                nullptr);

            PreShade(
                width, height,
                0,
                outputSurf);

            ShadeAO(width, height);

            ShadeMissAO(width, height);
        }

        onGather(outputSurf, width, height, maxSamples);

        checkCudaErrors(cudaDeviceSynchronize());

        m_frame++;
    }
}
