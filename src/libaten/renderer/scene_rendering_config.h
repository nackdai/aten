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
    };
}
