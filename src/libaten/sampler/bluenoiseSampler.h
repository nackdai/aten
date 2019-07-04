#pragma once

#include <memory>
#include <vector>

#include "types.h"
#include "math/vec2.h"
#include "texture/texture.h"
#include "sampler/samplerinterface.h"

namespace aten {
    class BlueNoiseSampler AT_INHERIT(sampler) {
    public:
        BlueNoiseSampler() = default;
        ~BlueNoiseSampler() {}

    public:
        void init(
            uint32_t x, uint32_t y, uint32_t frame,
            uint32_t maxBounceNum,
            uint32_t shadowRayNum)  // TODO Only for NEE
        {
            AT_ASSERT(!m_noise.empty());

            auto resW = m_noise[0]->width();
            auto resH = m_noise[0]->height();

            m_seed = (x % resW) << 0;
            m_seed |= (y % resH) << 10;
            m_seed |= (frame) << 20;

            m_noiseResW = resW;
            m_noiseResH = resH;
            m_noiseTexNum = m_noise.size();

            // NOTE
            // (1) Generate Ray : 2
            // Bounce : N
            //  Shadow Ray : 2
            //   (2) Sample light : 1
            //   (3) Sample point on light : 4 (for area light)
            //  (4) Smple BRDF direction : 2
            //  (5) Russian roulette : 1
            m_maxSampleNum = 2 + maxBounceNum * (2 * (1 + 4) + 2 + 1);
        }

        void registerNoiseTexture(const std::shared_ptr<texture>& noisetex)
        {
            m_noise.push_back(std::move(noisetex));
        }

        virtual real nextSample() final
        {
            float r = sample();
            return r;
        }

        aten::vec2 nextSample2D()
        {
            aten::vec2 r;
            r.x = sample();
            r.y = sample();
            return std::move(r);
        }

    private:
        real sample()
        {
            auto x = m_seed;
            auto y = m_seed >> 10;
            auto z = m_seed >> 20;
            z = (z * m_maxSampleNum + m_sampleCount);

            x &= m_noiseResW - 1;
            y &= m_noiseResH - 1;
            z &= m_noiseTexNum - 1;

            AT_ASSERT(z < m_noise.size());

            real u = x / (real)m_noiseResW;
            real v = y / (real)m_noiseResH;

            m_sampleCount++;

            auto noise = m_noise[z]->at(u, v);

            return aten::cmpMin(noise.x, 0.9999999999999f);
        }

    private:
        uint32_t m_seed{ 0 };

        uint32_t m_maxSampleNum{ 0 };
        
        uint16_t m_noiseResW{ 0 };
        uint16_t m_noiseResH{ 0 };
        uint16_t m_noiseTexNum{ 0 };
        uint16_t m_sampleCount{ 0 };

        std::vector<std::shared_ptr<texture>> m_noise;
    };
}
