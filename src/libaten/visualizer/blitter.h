#pragma once

#include "visualizer/pixelformat.h"
#include "visualizer/visualizer.h"

#define AT_EDIT_BLITTER_PARAM_RANGE(editor, param, min, max)    editor->Edit(#param, param##_, min, max)

namespace aten {
    class BlitterParameterEditor {
    public:
        BlitterParameterEditor() = default;
        virtual ~BlitterParameterEditor() = default;

        BlitterParameterEditor(const BlitterParameterEditor&) = delete;
        BlitterParameterEditor(BlitterParameterEditor&&) = delete;
        BlitterParameterEditor& operator=(const BlitterParameterEditor&) = delete;
        BlitterParameterEditor& operator=(BlitterParameterEditor&&) = delete;

        virtual bool Edit(std::string_view name, float& param, float _min = 0.0F, float _max = 1.0F) const { return false; }
    };


    class Blitter : public visualizer::PostProc {
    public:
        Blitter() {}
        virtual ~Blitter() {}

    public:
        virtual void PrepareRender(
            const void* pixels,
            bool revert) override;

        virtual bool Edit(const BlitterParameterEditor* editor) { return false; }

        virtual PixelFormat inFormat() const override
        {
            return PixelFormat::rgba32f;
        }
        virtual PixelFormat outFormat() const override
        {
            return PixelFormat::rgba32f;
        }
    };

}
