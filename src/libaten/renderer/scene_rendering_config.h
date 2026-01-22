#pragma once

#include "types.h"
#include "npr/feature_line_config.h"

namespace aten
{
    struct BackgroundResource {
        aten::vec3 bg_color{ 0.0F };

        int32_t envmap_tex_idx{ -1 };
        float avgIllum{ 1.0F };
        float multiplyer{ 1.0F };
        bool enable_env_map{ true };
    };

    struct SceneRenderingConfig {
        /**
         * @brief Control if alpha blending is enabled.
         */
        bool enable_alpha_blending{ false };

        /**
         * @brief Configuration to render feature line.
         */
        FeatureLineConfig feature_line;

        /**
         * @brief Configuration for minimum value how close hit is acceptable.
         *
         * If the value is negative, the default value is used.
         */
        float bvh_hit_min{ -1.0F };

        /**
        * @brief Epsilon bias for traversing shadow ray in medium.
        */
        float epsilon_bias_for_traversing_shadow_ray_in_medium{ 1e-3F };

        BackgroundResource bg;
    };
}
