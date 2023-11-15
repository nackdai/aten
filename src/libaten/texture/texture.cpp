#include <string>

#include "stb_image_write.h"

#include "texture/texture.h"
#include "visualizer/atengl.h"
#include "visualizer/shader.h"
#include "misc/color.h"

namespace aten
{
    texture::texture(int32_t width, int32_t height, uint32_t channels, std::string_view name)
    {
        init(width, height, channels);
        if (!name.empty()) {
            m_name = name;
        }
    }

    texture::~texture()
    {
        releaseAsGLTexture();
    }

    std::shared_ptr<texture> texture::create(
        int32_t width, int32_t height,
        uint32_t channels,
        std::string_view name)
    {
        auto ret = std::make_shared<texture>(width, height, channels, name);
        AT_ASSERT(ret);

        return ret;
    }

    void texture::init(int32_t width, int32_t height, uint32_t channels)
    {
        if (m_colors.empty()) {
            width_ = width;
            height_ = height;
            m_channels = channels;

            m_size = height * width;

            m_colors.resize(width * height);
        }
    }

    bool texture::initAsGLTexture()
    {
        if (m_gltex == 0) {
            AT_VRETURN(width_ > 0, false);
            AT_VRETURN(height_ > 0, false);
            AT_VRETURN(m_colors.size() > 0, false);

            CALL_GL_API(::glGenTextures(1, &m_gltex));
            AT_VRETURN(m_gltex > 0, false);

            CALL_GL_API(glBindTexture(GL_TEXTURE_2D, m_gltex));

            CALL_GL_API(glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA32F,
                width_, height_,
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

    bool texture::initAsGLTexture(int32_t width, int32_t height)
    {
        width_ = width;
        height_ = height;

        AT_VRETURN(width_ > 0, false);
        AT_VRETURN(height_ > 0, false);

        CALL_GL_API(::glGenTextures(1, &m_gltex));
        AT_VRETURN(m_gltex > 0, false);

        CALL_GL_API(glBindTexture(GL_TEXTURE_2D, m_gltex));

        CALL_GL_API(glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA32F,
            width_, height_,
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
        // shaderはバインドされていること.

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
        int32_t& width,
        int32_t& height,
        int32_t& channel,
        std::vector<vec4>& dst) const
    {
        if (m_gltex > 0) {
            width = width_;
            height = height_;
            channel = m_channels;

            dst.resize(width_ * height_);

            int32_t bufsize = width_ * height_ * sizeof(float) * 4;

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
        AT_VRETURN(width_ == rhs.width_, false);
        AT_VRETURN(height_ == rhs.height_, false);
        AT_VRETURN(m_colors.size() == rhs.m_colors.size(), false);

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
        for (int32_t y = 0; y < height_; y++) {
            for (int32_t x = 0; x < width_; x++) {
                int32_t idx = y * width_ + x;

                m_colors[idx] += rhs.m_colors[idx];
            }
        }

        return true;
    }

    bool texture::exportAsPNG(const std::string& filename)
    {
        using ScreenShotImageType = TColor<uint8_t, 3>;

        std::vector<ScreenShotImageType> dst(width_ * height_);

        constexpr int32_t bpp = ScreenShotImageType::BPP;
        const int32_t pitch = width_ * bpp;

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
        for (int32_t y = 0; y < height_; y++) {
            for (int32_t x = 0; x < width_; x++) {
                int32_t yy = height_ - 1 - y;

                dst[yy * width_ + x].r() = (uint8_t)aten::clamp(m_colors[y * width_ + x].x * real(255), real(0), real(255));
                dst[yy * width_ + x].g() = (uint8_t)aten::clamp(m_colors[y * width_ + x].y * real(255), real(0), real(255));
                dst[yy * width_ + x].b() = (uint8_t)aten::clamp(m_colors[y * width_ + x].z * real(255), real(0), real(255));
            }
        }

        auto ret = ::stbi_write_png(filename.c_str(), width_, height_, bpp, &dst[0], pitch);
        AT_ASSERT(ret > 0);

        return (ret > 0);
    }
}
