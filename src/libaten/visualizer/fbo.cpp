#include "visualizer/atengl.h"
#include "visualizer/fbo.h"

namespace aten {
    bool FBO::init(
        int32_t width,
        int32_t height,
        PixelFormat fmt,
        bool needDepth/*= false*/)
    {
        if (m_fbo > 0) {
            // TODO
            // Check size, format...

            return true;
        }

        CALL_GL_API(glGenFramebuffers(1, &m_fbo));

        m_tex.resize(m_num);

        CALL_GL_API(glGenTextures(m_num, &m_tex[0]));

        for (int32_t i = 0; i < m_num; i++) {
            CALL_GL_API(glBindTexture(GL_TEXTURE_2D, m_tex[i]));

            GLenum pixelfmt = 0;
            GLenum pixeltype = 0;
            GLenum pixelinternal = 0;

            getGLPixelFormat(
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

        m_width = width;
        m_height = height;
        m_fmt = fmt;

        if (needDepth) {
            CALL_GL_API(::glGenTextures(1, &m_depth));

            CALL_GL_API(glBindTexture(GL_TEXTURE_2D, m_depth));

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

    void FBO::bindAsTexture(uint32_t idx/*= 0*/)
    {
        AT_ASSERT(m_tex[idx] > 0);

        CALL_GL_API(glBindTexture(
            GL_TEXTURE_2D,
            m_tex[idx]));

        // Specify filter after binding!!!!!
        CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
        CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));
        CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    }

    void FBO::bindFBO(bool needDepth/*= false*/)
    {
        AT_ASSERT(isValid());

        CALL_GL_API(glBindFramebuffer(GL_FRAMEBUFFER, m_fbo));

        if (m_func) {
            m_func(&m_tex[0], m_num, m_comps);
        }
        else {
            if (m_comps.empty()) {
                for (int32_t i = 0; i < m_num; i++) {
                    CALL_GL_API(glFramebufferTexture2D(
                        GL_FRAMEBUFFER,
                        GL_COLOR_ATTACHMENT0 + i,
                        GL_TEXTURE_2D,
                        m_tex[i],
                        0));

                    m_comps.push_back(GL_COLOR_ATTACHMENT0 + i);
                }
            }
        }

        CALL_GL_API(glDrawBuffers(m_comps.size(), &m_comps[0]));

        if (needDepth) {
            AT_ASSERT(m_depth > 0);

            CALL_GL_API(::glFramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_DEPTH_ATTACHMENT,
                GL_TEXTURE_2D,
                m_depth,
                0));
        }

        //auto res = glCheckNamedFramebufferStatus(m_fbo, GL_FRAMEBUFFER);
        //AT_ASSERT(res == GL_FRAMEBUFFER_COMPLETE);
    }

    void FBO::asMulti(uint32_t num)
    {
        AT_ASSERT(num > 0);
        AT_ASSERT(m_fbo == 0);

        m_num = num;
    }

    void FBO::SaveToBuffer(std::vector<uint8_t>& dst, int32_t target_idx/*= 0*/)
    {
        const auto bpp = GetBytesPerPxiel(m_fmt);
        dst.resize(m_width * m_height * bpp);

        CALL_GL_API(::glBindFramebuffer(GL_FRAMEBUFFER, m_fbo));
        CALL_GL_API(::glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_tex[target_idx], 0));

        GLenum pixelfmt = 0;
        GLenum pixeltype = 0;
        GLenum pixelinternal = 0;

        getGLPixelFormat(
            m_fmt,
            pixelfmt, pixeltype, pixelinternal);

        CALL_GL_API(::glReadBuffer(GL_COLOR_ATTACHMENT0));
        CALL_GL_API(::glReadPixels(0, 0, m_width, m_height, pixelfmt, pixeltype, dst.data()));
    }
}
