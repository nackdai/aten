#pragma once

#include "types.h"
#include "visualizer/pixelformat.h"

#include <vector>
#include <functional>

namespace aten {
    class FBO {
    public:
        FBO() {}
        virtual ~FBO() {}

    public:
        bool init(
            int width, 
            int height, 
            PixelFormat fmt,
            bool needDepth = false);

        bool isValid() const
        {
            return (m_fbo > 0);
        }

        void bindAsTexture(uint32_t idx = 0);

        void bindFBO(bool needDepth = false);

        uint32_t getWidth() const
        {
            return m_width;
        }

        uint32_t getHeight() const
        {
            return m_height;
        }

        uint32_t getTexHandle(uint32_t idx = 0) const
        {
            return m_tex[idx];
        }

        uint32_t getHandle() const
        {
            return m_fbo;
        }

        void asMulti(uint32_t num);

        using FuncPrepareFbo = std::function<void(const uint32_t*, int, std::vector<uint32_t>&)>;
        void setPrepareFboFunction(FuncPrepareFbo func)
        {
            m_func = func;
        }

    protected:
        uint32_t m_fbo{ 0 };

        int m_num{ 1 };
        std::vector<uint32_t> m_tex;

        std::vector<uint32_t> m_comps;

        FuncPrepareFbo m_func{ nullptr };

        uint32_t m_depth{ 0 };

        PixelFormat m_fmt{ PixelFormat::rgba8 };
        uint32_t m_width{ 0 };
        uint32_t m_height{ 0 };
    };
}