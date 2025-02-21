#pragma once

#include "math/vec3.h"
#include "misc/color.h"
#include "visualizer/blitter.h"

namespace aten
{
    // sRGB OETF = Opto-Electronic Transfer Function
    class SRGBOptoElectronicTransferFunction : public Blitter {
    public:
        SRGBOptoElectronicTransferFunction() = default;
        ~SRGBOptoElectronicTransferFunction() = default;

        static constexpr const char* VertexShaderFile = "shader/fullscreen_vs.glsl";
        static constexpr const char* FragmentShaderFile = "shader/srgb_oetf_fs.glsl";

        bool Init(
            int width, int height,
            std::string_view base_path)
        {
            std::string vs(base_path);
            vs += SRGBOptoElectronicTransferFunction::VertexShaderFile;

            std::string fs(base_path);
            fs += SRGBOptoElectronicTransferFunction::FragmentShaderFile;

            return Blitter::init(width, height, vs, fs);
        }
    };
}
