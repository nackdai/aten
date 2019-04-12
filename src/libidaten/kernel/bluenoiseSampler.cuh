#pragma once

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"

#include "types.h"

namespace idaten {
    class BlueNoiseSampler {
    public:
        __device__ BlueNoiseSampler() {}
        __device__ ~BlueNoiseSampler() {};

    public:
        __device__ void init(
            uint32_t seed,
            uint16_t bounce,
            uint16_t resW,
            uint16_t resH,
            uint16_t num,
            cudaTextureObject_t noisetex)
        {
            m_seed = seed;
            m_noise = noisetex;

            m_dimension = 2 + 15 + 30 * bounce;
            m_noiseResW = resW;
            m_noiseResH = resH;
            m_noiseNum = num;
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
            return std::move(r);
        }

    private:
        AT_DEVICE_API float sample()
        {
            // http://developer.download.nvidia.com/CUDA/training/texture_webinar_aug_2011.pdf

            uint3 p = make_uint3(m_seed, m_seed >> 10, m_seed >> 20);
            p.z = (p.z * 13 + m_dimension);

            p.x &= m_noiseResW - 1;
            p.y &= m_noiseResH - 1;
            p.z &= m_noiseNum - 1;

            real u = p.x / (real)m_noiseResW;
            real v = p.y / (real)m_noiseResH;

            m_dimension++;

#ifdef __CUDA_ARCH__ 
            float4 ret = tex2DLayered<float4>(m_noise, u, v, p.z);
            return ret.x;
#else
            return real(1);
#endif
        }

    private:
        uint32_t m_seed{ 0 };
        
        uint16_t m_noiseResW{ 0 };
        uint16_t m_noiseResH{ 0 };
        uint16_t m_noiseNum{ 0 };
        uint16_t m_dimension{ 0 };

        cudaTextureObject_t m_noise;
    };
}
