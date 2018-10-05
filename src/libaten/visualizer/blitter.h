#pragma once

#include "visualizer/pixelformat.h"
#include "visualizer/visualizer.h"

namespace aten {
    class Blitter : public visualizer::PostProc {
    public:
        Blitter() {}
        virtual ~Blitter() {}

    public:
        virtual void prepareRender(
            const void* pixels,
            bool revert) override;

        virtual PixelFormat inFormat() const override
        {
            return PixelFormat::rgba32f;
        }
        virtual PixelFormat outFormat() const override
        {
            return PixelFormat::rgba32f;
        }

        void setIsRenderRGB(bool f)
        {
            m_isRenderRGB = f;
        }
        bool isRenderRGB() const
        {
            return m_isRenderRGB;
        }

    private:
        bool m_isRenderRGB{ true };
    };

}