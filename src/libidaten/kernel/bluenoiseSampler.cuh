#pragma once

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"

#include "types.h"
#include "math/vec2.h"

namespace idaten {
    class BlueNoiseSamplerGPU {
    public:
        __device__ BlueNoiseSamplerGPU() {}
        __device__ ~BlueNoiseSamplerGPU() {};

    public:
        __device__ static inline uint32_t makeSeed(
            uint32_t x, uint32_t y, uint32_t frame,
            uint32_t resW,
            uint32_t resH)
        {
            auto seed = (x % resW) << 0;
            seed |= (y % resH) << 10;
            seed |= (frame) << 20;
            return seed;
        }

        __device__ static inline uint32_t computeMaxSampleNum(uint32_t maxBounceNum)
        {
            // NOTE
            // (1) Generate Ray : 2
            // Bounce : N
            //  Shadow Ray : 2
            //   (2) Sample light : 1
            //   (3) Sample point on light : 4 (for area light)
            //  (4) Smple BRDF direction : 2
            //  (5) Russian roulette : 1
            auto maxSampleNum = 2 + maxBounceNum * (2 * (1 + 4) + 2 + 1);
            return maxSampleNum;
        }

        __device__ void init(
            uint32_t x, uint32_t y, uint32_t frame,
            uint32_t maxBounceNum,
            uint32_t shadowRayNum,  // TODO Only for NEE.
            uint32_t resW,
            uint32_t resH,
            uint16_t noiseTexNum,
            cudaTextureObject_t noisetex)
        {
            m_seed = makeSeed(x, y, frame, resW, resH);

            m_noise = noisetex;

            m_noiseResW = resW;
            m_noiseResH = resH;
            m_noiseTexNum = noiseTexNum;

            m_maxSampleNum = computeMaxSampleNum(maxBounceNum);
        }

        __device__ void init(
            uint32_t seed,
            uint32_t maxBounceNum,
            uint32_t shadowRayNum,  // TODO Only for NEE.
            uint32_t resW,
            uint32_t resH,
            uint16_t noiseTexNum,
            cudaTextureObject_t noisetex)
        {
            m_seed = seed;

            m_noise = noisetex;

            m_noiseResW = resW;
            m_noiseResH = resH;
            m_noiseTexNum = noiseTexNum;

            m_maxSampleNum = computeMaxSampleNum(maxBounceNum);
        }

        AT_DEVICE_API real nextSample()
        {
            float r = sample();
            return r;
        }

        AT_DEVICE_API aten::vec2 nextSample2D()
        {
            aten::vec2 r;
            r.x = sample();
            r.y = sample();
            return r;
        }

    private:
        AT_DEVICE_API float sample()
        {
            // http://developer.download.nvidia.com/CUDA/training/texture_webinar_aug_2011.pdf

            uint3 p = make_uint3(m_seed, m_seed >> 10, m_seed >> 20);
            p.z = (p.z * m_maxSampleNum + m_sampleCount);

            p.x &= m_noiseResW - 1;
            p.y &= m_noiseResH - 1;
            p.z &= m_noiseTexNum - 1;

            real u = p.x / (real)m_noiseResW;
            real v = p.y / (real)m_noiseResH;

            m_sampleCount++;

#ifdef __CUDA_ARCH__
            float4 ret = tex2DLayered<float4>(m_noise, u, v, p.z);
            return aten::cmpMin(ret.x, 0.9999999999999f);
#else
            return real(1.0f);
#endif
        }

    private:
        uint32_t m_seed{ 0 };

        uint32_t m_maxSampleNum{ 0 };

        uint16_t m_noiseResW{ 0 };
        uint16_t m_noiseResH{ 0 };
        uint16_t m_noiseTexNum{ 0 };
        uint16_t m_sampleCount{ 0 };

        cudaTextureObject_t m_noise;
    };
}
