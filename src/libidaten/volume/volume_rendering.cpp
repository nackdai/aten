#include "volume/volume_rendering.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

namespace idaten
{
    void VolumeRendering::OnRender(
        int32_t width, int32_t height,
        int32_t maxSamples,
        int32_t maxBounce,
        cudaSurfaceObject_t outputSurf)
    {
        static const int32_t rrBounce = 3;

        // Set bounce count to 1 forcibly, AOV rendering mode.
        maxBounce = (m_mode == Mode::AOV ? 1 : maxBounce);

        auto time = AT_NAME::timer::getSystemTime();

        for (int32_t i = 0; i < maxSamples; i++) {
            int32_t seed = time.milliSeconds;
            //int32_t seed = 0;

            generatePath(
                width, height,
                m_mode == Mode::AOV,
                i, maxBounce,
                seed);

            InitPathsForVolumeRendering(width, height);

            int32_t bounce = 0;

            while (true) {
                onHitTest(
                    width, height,
                    bounce);

                missShade(width, height, bounce);

                int32_t hitcount = 0;
                m_compaction.compact(
                    m_hitidx,
                    m_hitbools);

                //AT_PRINTF("%d\n", hitcount);

                onShade(
                    outputSurf,
                    width, height,
                    i,
                    bounce, rrBounce, maxBounce);

                const auto is_termnated = IsAllPathsTerminated(width, height, bounce);

                if (is_termnated) {
                    break;
                }

                bounce++;
            }
        }

        onGather(outputSurf, width, height, maxSamples);
    }
}
