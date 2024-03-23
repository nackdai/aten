#pragma once

#include "light/light_parameter.h"

namespace AT_NAME {
    /**
     * Reservoir.
     */
    struct Reservoir {
        float w_sum{ 0.0f };    ///< Sum of weights.
        int32_t M{ 0 };         ///< The number of samples seen so far.
        int32_t y{ -1 };        ///< The output sample.
        float W{ 0.0f };        ///< The stochastic weight for the generated sample y.
        float target_pdf_of_y{ 0.0f };   ///< The target PDF of generated sample y.

        aten::LightSampleResult light_sample_;

        /**
         * @brief Clear reservoir.
         */
        AT_HOST_DEVICE_API void clear()
        {
            w_sum = 0.0f;
            M = 0;
            y = -1;
            target_pdf_of_y = 0.0f;
            W = 0.0f;
        }

        /**
         * @brief Check if reservoir has valid sample.
         * @return If reservoir has valid sample, returns true, Otherwise, returns false.
         */
        AT_HOST_DEVICE_API bool IsValid() const
        {
            return y >= 0;
        }

        /**
         * @brief Update content of reservoir.
         * @param[in] light_sample
         * @param[in] sample sample.
         * @param[in] weight Weight of sample.
         * @param[in] m Number of samples seen so far.
         * @param[in] u Ramdom value if sample is adopted.
         * @return If sample is adopted, returns true. Otherwise, returns false.
         */
        AT_HOST_DEVICE_API bool update(
            const aten::LightSampleResult& light_sample,
            int32_t sample, float weight, int32_t m, float u)
        {
            w_sum += weight;
            bool is_accepted = u < weight / w_sum;
            if (is_accepted) {
                light_sample_ = light_sample;
                y = sample;
            }
            M += m;
            return is_accepted;
        }

        /**
         * @brief Update content of reservoir.
         * @param[in] light_sample
         * @param[in] sample sample.
         * @param[in] weight Weight of sample.
         * @param[in] u Ramdom value if sample is adopted.
         * @return If sample is adopted, returns true. Otherwise, returns false.
         */
        AT_HOST_DEVICE_API bool update(
            const aten::LightSampleResult& light_sample,
            int32_t sample, float weight, float u)
        {
            return update(light_sample, sample, weight, 1, u);
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

        aten::vec3 point_to_light;
        float v;

        aten::vec3 p;
        float pre_sampled_r;

        AT_HOST_DEVICE_API void clear()
        {
            nml.x = nml.y = nml.z = 0.0f;

            is_voxel = false;
            mtrl_idx = -1;

            wi.x = wi.y = wi.z = 0.0f;

            u = v = 0.0f;

            pre_sampled_r = 0.0f;
        }

        AT_HOST_DEVICE_API bool isMtrlValid() const
        {
            return mtrl_idx >= 0;
        }
    };
}
