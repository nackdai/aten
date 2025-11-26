#include "gradient_textue_editor.h"

bool GradientTextureEditor::Display()
{
    ImGG::Settings settings{};

    settings.gradient_width = 500.f;
    settings.gradient_height = 40.f;
    settings.horizontal_margin = 10.f;

    /// Distance under the gradient bar to delete a mark by dragging it down.
    /// This behaviour can also be disabled with the Flag::NoDragDowntoDelete.
    settings.distance_to_delete_mark_by_dragging_down = 80.f;

    settings.flags = ImGG::Flag::None;

    settings.color_edit_flags = ImGuiColorEditFlags_None;

    /// Controls how the new mark color is chosen.
    /// If true, the new mark color will be a random color,
    /// otherwise it will be the one that the gradient already has at the new mark position.
    settings.should_use_a_random_color_for_the_new_marks = false;

    auto result = gradient_widget_.widget("", settings);
    result |= ImGG::interpolation_mode_widget(
        "Interpolation Mode",
        &gradient_widget_.gradient().interpolation_mode());

    return result;
}
