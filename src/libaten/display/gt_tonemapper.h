#pragma once

#include "math/vec3.h"
#include "misc/color.h"
#include "visualizer/blitter.h"

namespace aten
{
    // Tone mapping from Gran Turismo 7.
    class GTTonemapper : public Blitter {
    public:
        GTTonemapper() = default;
        ~GTTonemapper() = default;

        static constexpr const char* VertexShaderFile = "shader/fullscreen_vs.glsl";
        static constexpr const char* FragmentShaderFile = "shader/gt_tonemapper_fs.glsl";

        bool Init(
            int width, int height,
            std::string_view base_path);

        virtual void PrepareRender(
            const void* pixels,
            bool revert) override final;

        bool Edit(const BlitterParameterEditor* editor) override;

    private:
        // Where to end the range of toe = Where to start the range of linear.
        AT_DEFINE_SHADER_PARAMETER(float, end_of_toe, 0.22F);

        // To control contrast.
        AT_DEFINE_SHADER_PARAMETER(float, contrast_param, 1.0F);

        // Max monitor luminance. 100[nit] = 1.0
        AT_DEFINE_SHADER_PARAMETER(float, max_monitor_luminance, 1.0F);

        // The range of linear.
        AT_DEFINE_SHADER_PARAMETER(float, range_of_linear, 0.4F);
    };
}
