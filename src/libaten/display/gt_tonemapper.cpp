#include "visualizer/atengl.h"
#include "display/gt_tonemapper.h"

namespace aten
{
    bool GTTonemapper::Init(
        int width, int height,
        std::string_view base_path)
    {
        std::string vs(base_path);
        vs += GTTonemapper::VertexShaderFile;

        std::string fs(base_path);
        fs += GTTonemapper::FragmentShaderFile;

        return Blitter::init(width, height, vs, fs);
    }

    void GTTonemapper::prepareRender(
        const void* pixels,
        bool revert)
    {
        Blitter::prepareRender(pixels, revert);

        auto h_end_of_toe = getHandle(name_end_of_toe);
        CALL_GL_API(::glUniform1f(h_end_of_toe, end_of_toe_));

        auto h_contrast_param = getHandle(name_contrast_param);
        CALL_GL_API(::glUniform1f(h_contrast_param, contrast_param_));

        auto h_max_monitor_luminance = getHandle(name_max_monitor_luminance);
        CALL_GL_API(::glUniform1f(h_max_monitor_luminance, max_monitor_luminance_));

        auto h_range_of_linear = getHandle(name_range_of_linear);
        CALL_GL_API(::glUniform1f(h_range_of_linear, range_of_linear_));
    }

    bool GTTonemapper::Edit(const BlitterParameterEditor* editor)
    {
        bool is_updated = false;

        is_updated |= AT_EDIT_BLITTER_PARAM_RANGE(editor, max_monitor_luminance, 0.0F, 30.0F);

        return is_updated;
    }
}
