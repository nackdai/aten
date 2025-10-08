#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "visualizer/atengl.h"
#include "visualizer/visualizer.h"
#include "math/vec3.h"

namespace aten
{
    PixelFormat visualizer::getPixelFormat()
    {
        return m_fmt;
    }

    static GLuint CreateTexture(int32_t width, int32_t height, PixelFormat fmt)
    {
        GLuint tex = 0;

        CALL_GL_API(::glGenTextures(1, &tex));
        AT_ASSERT(tex != 0);

        CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, tex));

        GLenum pixelfmt = 0;
        GLenum pixeltype = 0;
        GLenum pixelinternal = 0;

        GetGLPixelFormat(
            fmt,
            pixelfmt, pixeltype, pixelinternal);

        CALL_GL_API(::glTexImage2D(
            GL_TEXTURE_2D,
            0,
            pixelinternal,
            width, height,
            0,
            pixelfmt,
            pixeltype,
            NULL));

        CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

        CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
        CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));

        CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, 0));

        return tex;
    }

    uint32_t visualizer::GetGLTextureHandle()
    {
        return m_tex;
    }

    std::shared_ptr<visualizer> visualizer::init(int32_t width, int32_t height)
    {
        auto ret = std::make_shared<visualizer>();

        ret->m_tex = CreateTexture(width, height, ret->m_fmt);
        AT_VRETURN(ret->m_tex != 0, nullptr);

        ret->width_ = width;
        ret->height_ = height;

        return ret;
    }

    void visualizer::addPreProc(PreProc* preproc)
    {
        m_preprocs.push_back(preproc);
    }

    bool visualizer::addPostProc(PostProc* postproc)
    {
        if (m_postprocs.size() > 0) {
            // Create fbo to connect between post-processes.
            auto idx = m_postprocs.size() - 1;
            auto* prevPostproc = m_postprocs[idx];
            auto outFmt = prevPostproc->outFormat();

            // Check in-out format.
            auto inFmt = postproc->inFormat();
            AT_VRETURN(inFmt == outFmt, false);

            auto& fbo = prevPostproc->getFbo();

            // Create FBO.
            AT_VRETURN(fbo.init(width_, height_, outFmt), false);
        }

        postproc->setVisualizer(this);
        m_postprocs.push_back(postproc);

        return true;
    }

    const void* visualizer::doPreProcs(const vec4* pixels)
    {
        const void* textureimage = pixels;

        if (!m_preprocs.empty()) {
            uint32_t bufpos = 0;
            const vec4* src = (const vec4*)textureimage;
            vec4* dst = nullptr;

            for (int32_t i = 0; i < m_preprocs.size(); i++) {
                auto& buf = m_preprocBuffer[bufpos];
                if (buf.empty()) {
                    buf.resize(width_ * height_);
                }
                dst = &buf[0];

                (*m_preprocs[i])(src, width_, height_, dst);

                src = dst;
                bufpos = 1 - bufpos;
            }

            textureimage = src;
        }

        return textureimage;
    }

    const void* visualizer::convertTextureData(const void* textureimage)
    {
        // If type is double, convert double/rgb to float/rgba.
        // If type is float, convert rgb to rgba.
        if (m_tmp.empty()) {
            m_tmp.resize(width_ * height_);
        }

        const vec4* src = (const vec4*)textureimage;

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
        for (int32_t y = 0; y < height_; y++) {
            for (int32_t x = 0; x < width_; x++) {
                int32_t pos = y * width_ + x;

                auto& s = src[pos];
                auto& d = m_tmp[pos];

                d.r() = (float)s.x;
                d.g() = (float)s.y;
                d.b() = (float)s.z;
                d.a() = (float)s.w;
            }
        }

        textureimage = &m_tmp[0];

        return textureimage;
    }

    void visualizer::renderPixelData(
        const vec4* pixels,
        bool revert)
    {
        // Do pre processes.
        const void* textureimage = doPreProcs(pixels);

        CALL_GL_API(::glActiveTexture(GL_TEXTURE0));

        CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, m_tex));

        // Converte texture data double->float, rgb->rgba.
        textureimage = convertTextureData(textureimage);

        GLenum pixelfmt = 0;
        GLenum pixeltype = 0;
        GLenum pixelinternal = 0;

        GetGLPixelFormat(
            m_fmt,
            pixelfmt, pixeltype, pixelinternal);

        CALL_GL_API(::glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0, 0,
            width_, height_,
            pixelfmt,
            pixeltype,
            textureimage));

        ApplyPostProcesses(pixels, revert);
    }

    void visualizer::ApplyPostProcesses(
        const vec4* pixels,
        bool revert) const
    {
        // Keep how many post processes are enabled.
        int32_t count_enabled_post_procs = 0;

        for (const auto& post_proc : m_postprocs) {
            count_enabled_post_procs += post_proc->IsEnabled() ? 1 : 0;
        }

        bool willRevert = revert;

        // Count the current activated post process count.
        int32_t curr_activated_post_proc_count = 0;

        // Keep the previous post process.
        PostProc* prev_post_proc = nullptr;

        for (const auto& post_proc : m_postprocs) {
            if (!post_proc->IsEnabled()) {
                continue;
            }

            if (curr_activated_post_proc_count > 0) {
                auto& fbo = prev_post_proc->getFbo();

                CALL_GL_API(::glActiveTexture(GL_TEXTURE0));

                // Set FBO as source texture.
                fbo.BindAsTexture();
            }

            curr_activated_post_proc_count++;

            post_proc->PrepareRender(prev_post_proc, pixels, willRevert);

            if (curr_activated_post_proc_count == count_enabled_post_procs) {
                // We can assume that this post process is the last one and needs to bind with the default FBO.
                CALL_GL_API(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
            }
            else {
                auto& fbo = post_proc->getFbo();

                if (fbo.IsValid()) {
                    // Set FBO.
                    fbo.BindFBO();
                }
                else {
                    // Set default frame buffer.
                    CALL_GL_API(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
                }
            }

            CALL_GL_API(::glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));

            // It's fine to revert only once.
            willRevert = false;

            prev_post_proc = post_proc;
        }
    }

    void visualizer::render(bool revert)
    {
        RenderGLTextureByGLTexId(m_tex, revert);
    }

    void visualizer::renderGLTexture(aten::texture* tex, bool revert)
    {
        RenderGLTextureByGLTexId(
            tex->getGLTexHandle(), revert,
            [&tex] {
                tex->SetFilterAndAddressModeAsGLTexture();
            }
        );
    }

    void visualizer::RenderGLTextureByGLTexId(
        uint32_t gltex, bool revert,
        std::function<void()> OnSetTextureFilterAndAddress/*= nullptr*/)
    {
        // This API uses rendered OpenGL texture resource by GPGPU directly.
        // So, do not handle pixel data pointer directly.

        CALL_GL_API(::glActiveTexture(GL_TEXTURE0));

        CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, gltex));

        if (OnSetTextureFilterAndAddress) {
            OnSetTextureFilterAndAddress();
        }
#if 0
        else {
            // Specify filter after binding!!!!!
            CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
            CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));
            CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
            CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        }
#endif

        ApplyPostProcesses(nullptr, revert);
    }

    void visualizer::clear()
    {
        static const uint8_t clearclr_uc[4] = { 0, 0, 0, 0 };
        static const uint16_t clearclr_us[4] = { 0, 0, 0, 0 };
        static const uint32_t clearclr_ui[4] = { 0, 0, 0, 0 };

        GLenum pixelfmt = 0;
        GLenum pixeltype = 0;
        GLenum pixelinternal = 0;

        GetGLPixelFormat(
            m_fmt,
            pixelfmt, pixeltype, pixelinternal);

        const void* clearclr = nullptr;

        switch (m_fmt) {
        case PixelFormat::rgba8:
            clearclr = clearclr_uc;
            break;
        case PixelFormat::rgba32f:
            clearclr = clearclr_ui;
            break;
        case PixelFormat::rgba16f:
            clearclr = clearclr_us;
            break;
        default:
            AT_ASSERT(false);
            break;
        }

        CALL_GL_API(::glClearTexImage(
            m_tex,
            0,
            pixelfmt, pixeltype,
            clearclr));
    }

    void visualizer::takeScreenshot(std::string_view filename)
    {
        takeScreenshot(filename, width_, height_);
    }

    void visualizer::takeScreenshot(std::string_view filename, int32_t width, int32_t height)
    {
        CALL_GL_API(::glFlush());
        CALL_GL_API(::glFinish());

        using ScreenShotImageType = TColor<uint8_t, 3>;

        std::vector<ScreenShotImageType> tmp(width * height);

        CALL_GL_API(::glBindFramebuffer(GL_READ_FRAMEBUFFER, 0));
        CALL_GL_API(::glNamedFramebufferReadBuffer(0, GL_BACK));

        CALL_GL_API(::glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, &tmp[0]));

        // up-side-down.
        std::vector<ScreenShotImageType> dst(width * height);

        constexpr size_t bpp = ScreenShotImageType::BPP;
        const int32_t pitch = width * bpp;

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
        // NOTE
        // index variable in OpenMP 'for' statement must have signed integral type
        for (int32_t y = 0; y < height; y++) {
            auto yy = height - 1 - y;

            memcpy(
                &dst[yy * width],
                &tmp[y * width],
                pitch);
        }

        auto ret = ::stbi_write_png(filename.data(), width, height, bpp, dst.data(), pitch);
        AT_ASSERT(ret > 0);
    }

    void visualizer::getTextureData(
        uint32_t gltex,
        std::vector<TColor<uint8_t, 4>>& dst)
    {
        CALL_GL_API(::glFlush());
        CALL_GL_API(::glFinish());

        auto size = dst.size() * sizeof(uint8_t) * 4;

        CALL_GL_API(::glGetTextureImage(
            gltex,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            static_cast<GLsizei>(size),
            &dst[0]));
    }
}
