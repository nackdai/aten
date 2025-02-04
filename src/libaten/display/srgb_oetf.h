#pragma once

#include "math/vec3.h"
#include "misc/color.h"
#include "visualizer/blitter.h"

namespace aten
{
    // sRGB OETF = Opto-Electronic Transfer Function
    class sRGBOptoElectronicTransferFunction : public Blitter {
    public:
        sRGBOptoElectronicTransferFunction() = default;
        ~sRGBOptoElectronicTransferFunction() = default;

        static constexpr const char* VertexShaderFile = "shader/fullscreen_vs.glsl";
        static constexpr const char* FragmentShaderFile = "shader/srgb_oetf_fs.glsl";

        bool Init(
            int width, int height,
            std::string_view base_path)
        {
            std::string vs(base_path);
            vs += sRGBOptoElectronicTransferFunction::VertexShaderFile;

            std::string fs(base_path);
            fs += sRGBOptoElectronicTransferFunction::FragmentShaderFile;

            return Blitter::init(width, height, vs, fs);
        }
    };
}
