#include <string>

#include "stb_image_write.h"

#include "texture/texture.h"
#include "visualizer/atengl.h"
#include "visualizer/shader.h"
#include "misc/color.h"

namespace aten
{
    void texture::resetIdWhenAnyTextureLeave(aten::texture* tex)
    {
        tex->m_id = tex->m_listItem.currentIndex();
    }

    texture::texture()
    {
        m_listItem.init(this, resetIdWhenAnyTextureLeave);
    }

    texture::texture(uint32_t width, uint32_t height, uint32_t channels, const char* name/*= nullptr*/)
        : texture()
    {
        init(width, height, channels);
        if (name) {
            m_name = name;
        }
    }

    texture::~texture()
    {
        m_listItem.leave();

        releaseAsGLTexture();
    }

    texture* texture::create(uint32_t width, uint32_t height, uint32_t channels, const char* name)
    {
        texture* ret = new texture(width, height, channels, name);
        AT_ASSERT(ret);

        return ret;
    }

    void texture::init(uint32_t width, uint32_t height, uint32_t channels)
    {
        if (m_colors.empty()) {
            m_width = width;
            m_height = height;
            m_channels = channels;

            m_size = height * width;

            m_colors.resize(width * height);
        }
    }

    bool texture::initAsGLTexture()
    {
        if (m_gltex == 0) {
            AT_VRETURN(m_width > 0, false);
            AT_VRETURN(m_height > 0, false);
            AT_VRETURN(m_colors.size() > 0, false);

            CALL_GL_API(::glGenTextures(1, &m_gltex));
            AT_VRETURN(m_gltex > 0, false);

            CALL_GL_API(glBindTexture(GL_TEXTURE_2D, m_gltex));

            CALL_GL_API(glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA32F,
                m_width, m_height,
                0,
                GL_RGBA,
                GL_FLOAT,
                &m_colors[0]));

            CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
            CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

            CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
            CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));

            CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, 0));
        }

        return true;
    }

    bool texture::initAsGLTexture(int width, int height)
    {
        m_width = width;
        m_height = height;

        AT_VRETURN(m_width > 0, false);
        AT_VRETURN(m_height > 0, false);

        CALL_GL_API(::glGenTextures(1, &m_gltex));
        AT_VRETURN(m_gltex > 0, false);

        CALL_GL_API(glBindTexture(GL_TEXTURE_2D, m_gltex));

        CALL_GL_API(glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA32F,
            m_width, m_height,
            0,
            GL_RGBA,
            GL_FLOAT,
            nullptr));

        CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

        CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
        CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));

        CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, 0));

        return true;
    }

    void texture::bindAsGLTexture(uint8_t stage, shader* shd) const
    {
        bindAsGLTexture(m_gltex, stage, shd);
    }

    void texture::bindAsGLTexture(
        uint32_t gltex,
        uint8_t stage, shader* shd)
    {
        AT_ASSERT(gltex > 0);
        AT_ASSERT(shd);

        // NOTE
        // shader‚ÍƒoƒCƒ“ƒh‚³‚ê‚Ä‚¢‚é‚±‚Æ.

        std::string texuniform = std::string("s") + std::to_string(stage);
        auto handle = shd->getHandle(texuniform.c_str());
        if (handle >= 0) {
            CALL_GL_API(::glUniform1i(handle, stage));

            CALL_GL_API(::glActiveTexture(GL_TEXTURE0 + stage));

            CALL_GL_API(glBindTexture(GL_TEXTURE_2D, gltex));

            CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
            CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));

            CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
            CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));
        }
    }

    void texture::releaseAsGLTexture()
    {
        if (m_gltex > 0) {
            CALL_GL_API(::glDeleteTextures(1, &m_gltex));
            m_gltex = 0;
        }
    }

    void texture::clearAsGLTexture(const aten::vec4& clearColor)
    {
        if (m_gltex > 0) {
            const float clearclr[4] = {
                clearColor.x,
                clearColor.y,
                clearColor.z,
                clearColor.w,
            };

            CALL_GL_API(::glClearTexImage(
                m_gltex,
                0,
                GL_RGBA,
                GL_FLOAT,
                clearclr));
        }
    }

    void texture::getDataAsGLTexture(
        int& width,
        int& height,
        int& channel,
        std::vector<vec4>& dst) const
    {
        if (m_gltex > 0) {
            width = m_width;
            height = m_height;
            channel = m_channels;

            dst.resize(m_width * m_height);

            int bufsize = m_width * m_height * sizeof(float) * 4;

            CALL_GL_API(::glGetTextureImage(
                m_gltex,
                0,
                GL_RGBA,
                GL_FLOAT,
                bufsize,
                &dst[0]));
        }
    }

    bool texture::merge(const texture& rhs)
    {
        AT_VRETURN(m_width == rhs.m_width, false);
        AT_VRETURN(m_height == rhs.m_height, false);
        AT_VRETURN(m_colors.size() == rhs.m_colors.size(), false);

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
        for (int y = 0; y < m_height; y++) {
            for (int x = 0; x < m_width; x++) {
                int idx = y * m_width + x;

                m_colors[idx] += rhs.m_colors[idx];
            }
        }

        return true;
    }

    bool texture::exportAsPNG(const std::string& filename)
    {
        using ScreenShotImageType = TColor<uint8_t, 3>;

        std::vector<ScreenShotImageType> dst(m_width * m_height);

        static const int bpp = sizeof(ScreenShotImageType);
        const int pitch = m_width * bpp;

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
        for (int y = 0; y < m_height; y++) {
            for (int x = 0; x < m_width; x++) {
                int yy = m_height - 1 - y;

                dst[yy * m_width + x].r() = (uint8_t)aten::clamp(m_colors[y * m_width + x].x * real(255), real(0), real(255));
                dst[yy * m_width + x].g() = (uint8_t)aten::clamp(m_colors[y * m_width + x].y * real(255), real(0), real(255));
                dst[yy * m_width + x].b() = (uint8_t)aten::clamp(m_colors[y * m_width + x].z * real(255), real(0), real(255));
            }
        }

        auto ret = ::stbi_write_png(filename.c_str(), m_width, m_height, bpp, &dst[0], pitch);
        AT_ASSERT(ret > 0);

        return (ret > 0);
    }
}