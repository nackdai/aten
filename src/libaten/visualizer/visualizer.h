#pragma once

#include "defs.h"
#include "math/vec4.h"
#include "visualizer/pixelformat.h"
#include "visualizer/shader.h"
#include "visualizer/fbo.h"
#include "misc/value.h"
#include "misc/color.h"

namespace aten {
    class visualizer {
    private:
        visualizer() {}
        ~visualizer() {}

    public:
        class PreProc {
        protected:
            PreProc() {}
            virtual ~PreProc() {}

        public:
            virtual void operator()(
                const vec4* src,
                uint32_t width, uint32_t height,
                vec4* dst) = 0;

            virtual void setParam(Values& values) {}
        };

        class PostProc : public shader {
        protected:
            PostProc() {}
            virtual ~PostProc() {}

        public:
            virtual void prepareRender(
                const void* pixels,
                bool revert) override
            {
                shader::prepareRender(pixels, revert);
            }

            virtual PixelFormat inFormat() const = 0;
            virtual PixelFormat outFormat() const = 0;

            virtual void setParam(Values& values) {}

            virtual uint32_t getOutWidth() const
            {
                return m_width;
            }
            virtual uint32_t getOutHeight() const
            {
                return m_height;
            }

            virtual FBO& getFbo()
            {
                return m_fbo;
            }
            const FBO& getFbo() const
            {
                return m_fbo;
            }

            PostProc* getPrevPass()
            {
                return m_prevPass;
            }

            void prepareRender(
                PostProc* prevPass,
                const void* pixels,
                bool revert)
            {
                m_prevPass = prevPass;
                prepareRender(pixels, revert);
            }

        private:
            FBO m_fbo;
            PostProc* m_prevPass{ nullptr };
        };

    public:
        PixelFormat getPixelFormat();

        static uint32_t getTexHandle();
        static visualizer* init(int width, int height);

        void addPreProc(PreProc* preproc);

        bool addPostProc(PostProc* postproc);

        void render(
            const vec4* pixels,
            bool revert);

        void render(bool revert);
        void render(uint32_t gltex, bool revert);

        void clear();

        void takeScreenshot(const char* filename);

        static void takeScreenshot(const char* filename, uint32_t width, uint32_t height);

        static void getTextureData(
            uint32_t gltex,
            std::vector<TColor<uint8_t, 4>>& dst);

    private:
        const void* doPreProcs(const vec4* pixels);
        const void* convertTextureData(const void* textureimage);

    private:
        static visualizer* s_curVisualizer;

        uint32_t m_tex{ 0 };

        int m_width{ 0 };
        int m_height{ 0 };

        std::vector<TColor<float, 4>> m_tmp;

        const PixelFormat m_fmt{ PixelFormat::rgba32f };

        std::vector<visualizer::PreProc*> m_preprocs;
        std::vector<vec4> m_preprocBuffer[2];

        std::vector<visualizer::PostProc*> m_postprocs;
    };
}
