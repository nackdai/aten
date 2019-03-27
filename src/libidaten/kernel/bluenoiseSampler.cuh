#pragma once

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"

#include "types.h"
#include "kernel/idaten_sampler.h"

namespace idaten {
    class BlueNoiseSampler AT_INHERIT(sampler) {
    public:
        AT_DEVICE_API BlueNoiseSampler() {}
        AT_VIRTUAL(AT_DEVICE_API ~BlueNoiseSampler() {});

    public:
        AT_VIRTUAL_OVERRIDE_FINAL(AT_DEVICE_API void init(uint32_t seed, const void* data = nullptr))
        {
            AT_ASSERT(false);
            m_seed = seed;
            m_noise = reinterpret_cast<const float4*>(data);
        }

        void init(
            uint32_t seed,
            uint16_t bounce,
            uint16_t resW,
            uint16_t resH,
            uint16_t num,
            const float4* data)
        {
            m_seed = seed;
            m_noise = reinterpret_cast<const float4*>(data);

            m_dimension = 2 + 15 + 30 * bounce;
            m_noiseResW = resW;
            m_noiseResH = resH;
            m_noiseNum = num;
        }

        AT_VIRTUAL_OVERRIDE_FINAL(AT_DEVICE_API real nextSample())
        {
            float r = sample();
            return r;
        }

        AT_VIRTUAL_OVERRIDE_FINAL(AT_DEVICE_API aten::vec2 nextSample2D())
        {
            aten::vec2 r;
            r.x = sample();
            r.y = sample();
            return std::move(r);
        }

    private:
        AT_DEVICE_API float sample()
        {
            uint3 p = make_uint3(m_seed, m_seed >> 10, m_seed >> 20);
            p.z = (p.z * 13 + m_dimension);

            p.x &= m_noiseResW - 1;
            p.y &= m_noiseResH - 1;
            p.z &= m_noiseNum - 1;

            uint32_t size = m_noiseResW * m_noiseResH;
            auto i = p.y * m_noiseResW + p.x;

            auto pos = p.z * size + i;

            float ret = m_noise[pos].x;

            m_dimension++;

            return ret;
        }

    private:
        uint32_t m_seed{ 0 };
        
        uint16_t m_noiseResW{ 0 };
        uint16_t m_noiseResH{ 0 };
        uint16_t m_noiseNum{ 0 };
        uint16_t m_dimension{ 0 };

        const float4* m_noise{ nullptr };
    };
}
