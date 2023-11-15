#pragma once

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "aten4idaten.h"

namespace idaten {
    struct Reservoir {
        float w_sum_{ 0.0f };
        float sample_weight_{ 0.0f };
        uint32_t m_{ 0 };
        int32_t light_idx_{ 0 };
        float pdf_{ 0.0f };
        float target_density_{ 0.0f };
        aten::LightSampleResult light_sample_;

        __host__ __device__ void clear()
        {
            w_sum_ = 0.0f;
            sample_weight_ = 0.0f;
            m_ = 0;
            light_idx_ = -1;
            pdf_ = 0.0f;
            target_density_ = 0.0f;
        }

        __host__ __device__ bool IsValid() const
        {
            return light_idx_ >= 0;
        }

        __host__ __device__ bool update(
            const aten::LightSampleResult& light_sample,
            int32_t new_target_idx, float weight, uint32_t m, float u)
        {
            w_sum_ += weight;
            bool is_accepted = u < weight / w_sum_;
            if (is_accepted) {
                light_sample_ = light_sample;
                light_idx_ = new_target_idx;
                sample_weight_ = weight;
            }
            m_ += m;
            return is_accepted;
        }

        __host__ __device__ bool update(
            const aten::LightSampleResult& light_sample,
            int32_t new_target_idx, float weight, float u)
        {
            return update(light_sample, new_target_idx, weight, 1, u);
        }
    };
}
