#pragma once

#include "types.h"
#include "npr/feature_line_config.h"

namespace aten
{
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
         * If the value is netative, the default value is used.
         */
        float bvh_hit_min{ -1.0F };
    };
}
