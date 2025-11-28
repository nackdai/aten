#pragma once

#include <vector>

// NOTE:
// Add '#define IMGUI_DEFINE_MATH_OPERATORS' before including this file (or in imconfig.h) to access courtesy maths operators for ImVec2 and ImVec4.
#define IMGUI_DEFINE_MATH_OPERATORS

#include <imgui.h>
#include <imgui_gradient/imgui_gradient.hpp>

#include "math/vec4.h"

class GradientTextureEditor {
public:
    GradientTextureEditor() = default;
    ~GradientTextureEditor() = default;

    GradientTextureEditor(const GradientTextureEditor&) = delete;
    GradientTextureEditor(GradientTextureEditor&&) = delete;
    GradientTextureEditor& operator=(const GradientTextureEditor&) = delete;
    GradientTextureEditor& operator=(GradientTextureEditor&&) = delete;

    bool Display();

    void Read(
        aten::vec4* dst_1d,
        size_t width, size_t height);

private:
    ImGG::GradientWidget gradient_widget_;
};
