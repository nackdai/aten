#pragma once

#include "defs.h"
#include "visualizer/pixelformat.h"
#include "visualizer/shader.h"
#include "visualizer/fbo.h"

#include "math/vec4.h"
#include "misc/value.h"
#include "misc/color.h"
#include "image/texture.h"

namespace aten {
    class visualizer {
    public:
        visualizer() = default;
        ~visualizer() = default;

        class PreProc {
        protected:
            PreProc() {}
            virtual ~PreProc() {}

        public:
            virtual void operator()(
                const vec4* src,
                int32_t width, int32_t height,
                vec4* dst) = 0;

            virtual void setParam(Values& values) {}
        };

        class PostProc : public shader {
            friend class visualizer;

        protected:
            PostProc() {}
            virtual ~PostProc() {}

        public:
            virtual void PrepareRender(
                const void* pixels,
                bool revert) override
            {
                shader::PrepareRender(pixels, revert);
            }

            virtual PixelFormat inFormat() const = 0;
            virtual PixelFormat outFormat() const = 0;

            virtual void setParam(Values& values) {}

            virtual uint32_t getOutWidth() const
            {
                return width_;
            }
            virtual uint32_t getOutHeight() const
            {
                return height_;
            }

            virtual FBO& getFbo()
            {
                return fbo_;
            }
            const FBO& getFbo() const
            {
                return fbo_;
            }

            PostProc* getPrevPass()
            {
                return m_prevPass;
            }

            void PrepareRender(
                PostProc* prevPass,
                const void* pixels,
                bool revert)
            {
                m_prevPass = prevPass;
                PrepareRender(pixels, revert);
            }

            bool IsEnabled() const
            {
                return is_enabled_;
            }

            void SetIsEnabled(bool b)
            {
                is_enabled_ = b;
            }

        protected:
            void setVisualizer(aten::visualizer* visualizer)
            {
                m_visualizer = visualizer;
            }

            visualizer* getVisualizer()
            {
                return m_visualizer;
            }

            FBO fbo_;
            PostProc* m_prevPass{ nullptr };
            aten::visualizer* m_visualizer{ nullptr };

            bool is_enabled_{ true };
        };

    public:
        PixelFormat getPixelFormat();

        uint32_t GetGLTextureHandle();
        static std::shared_ptr<visualizer> init(int32_t width, int32_t height);

        void addPreProc(PreProc* preproc);

        bool addPostProc(PostProc* postproc);

        void renderPixelData(
            const vec4* pixels,
            bool revert);

        void render(bool revert);
        void renderGLTexture(aten::texture* tex, bool revert);

        void RenderGLTextureByGLTexId(
            uint32_t gltex, bool revert,
            std::function<void()> OnSetTextureFilterAndAddress = nullptr);

        void clear();

        void takeScreenshot(std::string_view filename);

        static void takeScreenshot(std::string_view filename, int32_t width, int32_t height);

        static void getTextureData(
            uint32_t gltex,
            std::vector<TColor<uint8_t, 4>>& dst);

    private:
        const void* doPreProcs(const vec4* pixels);
        const void* convertTextureData(const void* textureimage);

        void ApplyPostProcesses(
            const vec4* pixels,
            bool revert) const;

    private:
        uint32_t m_tex{ 0 };

        int32_t width_{ 0 };
        int32_t height_{ 0 };

        std::vector<TColor<float, 4>> m_tmp;

        const PixelFormat m_fmt{ PixelFormat::rgba32f };

        std::vector<visualizer::PreProc*> m_preprocs;
        std::vector<vec4> m_preprocBuffer[2];

        std::vector<visualizer::PostProc*> m_postprocs;
    };
}
