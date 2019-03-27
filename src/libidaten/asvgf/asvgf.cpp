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
        auto bytes = imgSize * sizeof(float4);
        auto num = noises.size();

        std::vector<float4> tmp(m_bluenoise.num());

        for (int n = 0; n < num; n++) {
            const auto noise = noises[n];
            const float4* data = reinterpret_cast<const float4*>(noise->colors());

            for (int i = 0; i < imgSize; i++) {
                int pos = n * imgSize + i;
                tmp[pos] = data[i];
            }
        }

        m_bluenoise.init(W * H * num);
        m_bluenoise.write(&tmp, bytes);

        return true;
    }
}