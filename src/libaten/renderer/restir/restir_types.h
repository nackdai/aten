#pragma once

#include "light/light_parameter.h"

namespace AT_NAME {
    struct Reservoir {
        float w_sum_{ 0.0f };
        uint32_t m_{ 0 };
        int32_t light_idx_{ 0 };
        float pdf_{ 0.0f };
        float target_density_{ 0.0f };
        aten::LightSampleResult light_sample_;

        AT_HOST_DEVICE_API void clear()
        {
            w_sum_ = 0.0f;
            m_ = 0;
            light_idx_ = -1;
            pdf_ = 0.0f;
            target_density_ = 0.0f;
        }

        AT_HOST_DEVICE_API bool IsValid() const
        {
            return light_idx_ >= 0;
        }

        AT_HOST_DEVICE_API bool update(
            const aten::LightSampleResult& light_sample,
            int32_t new_target_idx, float weight, uint32_t m, float u)
        {
            w_sum_ += weight;
            bool is_accepted = u < weight / w_sum_;
            if (is_accepted) {
                light_sample_ = light_sample;
                light_idx_ = new_target_idx;
            }
            m_ += m;
            return is_accepted;
        }

        AT_HOST_DEVICE_API bool update(
            const aten::LightSampleResult& light_sample,
            int32_t new_target_idx, float weight, float u)
        {
            return update(light_sample, new_target_idx, weight, 1, u);
        }
    };

    // NOTE
    // size of "bool" is expected as 1 byte.
    static_assert(sizeof(bool) == 1, "");

    struct ReSTIRInfo {
        aten::vec3 nml;
        int16_t mtrl_idx{ -1 };
        bool is_voxel{ false };
        uint8_t padding[3];

        aten::vec3 wi;
        float u;

        aten::vec3 throughput;
        float v;

        aten::vec3 p;
        float pre_sampled_r;

        AT_HOST_DEVICE_API void clear()
        {
            nml.x = nml.y = nml.z = 0.0f;

            is_voxel = false;
            mtrl_idx = -1;

            wi.x = wi.y = wi.z = 0.0f;
            throughput.x = throughput.y = throughput.z = 0.0f;

            u = v = 0.0f;

            pre_sampled_r = 0.0f;
        }

        AT_HOST_DEVICE_API bool isMtrlValid() const
        {
            return mtrl_idx >= 0;
        }
    };
}
