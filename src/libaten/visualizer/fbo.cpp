#include "visualizer/atengl.h"
#include "visualizer/fbo.h"

namespace aten {
    bool FBO::init(
        int32_t width,
        int32_t height,
        PixelFormat fmt,
        bool need_depth/*= false*/)
    {
        if (fbo_ > 0) {
            // TODO
            // Check size, format...

            return true;
        }

        CALL_GL_API(glGenFramebuffers(1, &fbo_));

        texture_handles_.resize(m_num);

        CALL_GL_API(glGenTextures(m_num, &texture_handles_[0]));

        for (int32_t i = 0; i < m_num; i++) {
            CALL_GL_API(glBindTexture(GL_TEXTURE_2D, texture_handles_[i]));

            GLenum pixelfmt = 0;
            GLenum pixeltype = 0;
            GLenum pixelinternal = 0;

            GetGLPixelFormat(
                fmt,
                pixelfmt, pixeltype, pixelinternal);

            CALL_GL_API(glTexImage2D(
                GL_TEXTURE_2D,
                0,
                pixelinternal,
                width, height,
                0,
                pixelfmt,
                pixeltype,
                nullptr));
        }

        width_ = width;
        height_ = height;
        pixel_fmt_ = fmt;

        if (need_depth) {
            CALL_GL_API(::glGenTextures(1, &depth_buffer_handle_));

            CALL_GL_API(glBindTexture(GL_TEXTURE_2D, depth_buffer_handle_));

            CALL_GL_API(::glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_DEPTH_COMPONENT,
                width, height,
                0,
                GL_DEPTH_COMPONENT,
                GL_UNSIGNED_INT,
                nullptr));
        }

        return true;
    }

    void FBO::BindAsTexture(uint32_t idx/*= 0*/)
    {
        AT_ASSERT(texture_handles_[idx] > 0);

        CALL_GL_API(glBindTexture(
            GL_TEXTURE_2D,
            texture_handles_[idx]));

        // Specify filter after binding!!!!!
        CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
        CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));
        CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    }

    void FBO::BindFBO(bool need_depth/*= false*/)
    {
        AT_ASSERT(IsValid());

        CALL_GL_API(glBindFramebuffer(GL_FRAMEBUFFER, fbo_));

        if (func_bind_fbo_) {
            func_bind_fbo_(texture_handles_, target_buffer_attachment_list_);
        }
        else {
            if (target_buffer_attachment_list_.empty()) {
                const auto tex_num = texture_handles_.size();
                for (size_t i = 0; i < tex_num; i++) {
                    CALL_GL_API(glFramebufferTexture2D(
                        GL_FRAMEBUFFER,
                        GL_COLOR_ATTACHMENT0 + i,
                        GL_TEXTURE_2D,
                        texture_handles_[i],
                        0));

                    target_buffer_attachment_list_.push_back(GL_COLOR_ATTACHMENT0 + i);
                }
            }
        }

        CALL_GL_API(glDrawBuffers(target_buffer_attachment_list_.size(), target_buffer_attachment_list_.data()));

        if (need_depth) {
            AT_ASSERT(depth_buffer_handle_ > 0);

            CALL_GL_API(::glFramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_DEPTH_ATTACHMENT,
                GL_TEXTURE_2D,
                depth_buffer_handle_,
                0));
        }

        //auto res = glCheckNamedFramebufferStatus(fbo_, GL_FRAMEBUFFER);
        //AT_ASSERT(res == GL_FRAMEBUFFER_COMPLETE);
    }

    void FBO::asMulti(uint32_t num)
    {
        AT_ASSERT(num > 0);
        AT_ASSERT(fbo_ == 0);

        m_num = num;
    }

    void FBO::SaveToBuffer(std::vector<uint8_t>& dst, int32_t target_idx/*= 0*/) const
    {
        const auto bpp = GetBytesPerPxiel(pixel_fmt_);
        dst.resize(width_ * height_ * bpp);

        CALL_GL_API(::glBindFramebuffer(GL_FRAMEBUFFER, fbo_));
        CALL_GL_API(::glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_handles_[target_idx], 0));

        GLenum pixelfmt = 0;
        GLenum pixeltype = 0;
        GLenum pixelinternal = 0;

        GetGLPixelFormat(
            pixel_fmt_,
            pixelfmt, pixeltype, pixelinternal);

        CALL_GL_API(::glReadBuffer(target_buffer_attachment_list_[target_idx]));
        CALL_GL_API(::glReadPixels(0, 0, width_, height_, pixelfmt, pixeltype, dst.data()));
    }
}
