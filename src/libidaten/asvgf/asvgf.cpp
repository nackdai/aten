#include "asvgf/asvgf.h"

namespace idaten
{
    bool AdvancedSVGFPathTracing::setBlueNoises(std::vector<aten::texture*>& noises)
    {
        const auto W = noises[0]->width();
        const auto H = noises[0]->height();

        // All noise texture have to be same size.
        {
            for (int i = 1; i < noises.size(); i++) {
                const auto n = noises[i];

                auto _w = n->width();
                auto _h = n->height();

                if (W != _w || H != _h) {
                    AT_ASSERT(false);
                    return false;
                }
            }
        }

        auto imgSize = W * H;
        auto bytes = imgSize * 4 * sizeof(float);  // 4 is count of channel (RGBA)
        auto num = noises.size();

        std::vector<float> tmp(m_bluenoise.num());

        for (int n = 0; n < num; n++) {
            const auto noise = noises[n];
            const float* data = reinterpret_cast<const float*>(noise->colors());

            for (int c = 0; c < 4; c++) {
                for (int i = 0; i < imgSize; i++) {
                    int pos = n * (imgSize * 4) + (i * 4) + c;
                    tmp[pos] = data[i * 4 + c];
                }
            }
        }

        m_bluenoise.init(W * H * 4 * num);
        m_bluenoise.write(&tmp, bytes);

        return true;
    }
}