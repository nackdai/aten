#include <string>

#include "stb_image_write.h"

#include "image/texture.h"
#include "visualizer/atengl.h"
#include "visualizer/shader.h"
#include "misc/color.h"

namespace aten
{
    texture::texture(int32_t width, int32_t height, int32_t channels, std::string_view name)
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
        int32_t channels,
        std::string_view name)
    {
        auto ret = std::make_shared<texture>(width, height, channels, name);
        AT_ASSERT(ret);

        return ret;
    }

    vec4 texture::AtByXY(int32_t x, int32_t y) const
    {
        uint32_t pos = y * width_ + x;

        const auto clr = value_[pos];

        // TODO
        // Note use alpha channel...
        uint32_t ch = std::min<uint32_t>(m_channels, 4);

        vec4 ret;

        if (ch >= 4) {
            ret[3] = clr[3];
        }
        if (ch >= 3) {
            ret[2] = clr[2];
        }
        if (ch >= 2) {
            ret[1] = clr[1];
        }
        if (ch >= 1) {
            ret[0] = clr[0];
        }

        return ret;
    }

    vec4 texture::at(float u, float v) const
    {
        int32_t iu = static_cast<int32_t>(u * (width_ - 1));
        int32_t iv = static_cast<int32_t>(v * (height_ - 1));

        // NOTE:
        // Wrap as repeat.
        const auto x = NormalizeToWrapRepeat(iu, width_ - 1);
        const auto y = NormalizeToWrapRepeat(iv, height_ - 1);

        return AtByXY(x, y);
    }

    vec4 texture::AtWithBilinear(float x, float y) const
    {
        float u = x * width_ - 0.5F;
        float v = y * height_ - 0.5F;
        const auto i = static_cast<int32_t>(aten::floor(u));
        const auto j = static_cast<int32_t>(aten::floor(v));
        u -= i;
        v -= j;

        int i0 = aten::max(0, aten::min(width_ - 1, i));
        int i1 = aten::max(0, aten::min(width_ - 1, i + 1));
        int j0 = aten::max(0, aten::min(height_ - 1, j));
        int j1 = aten::max(0, aten::min(height_ - 1, j + 1));

        // NOTE:
        // The following code is the same as expanding the following pseudo code.
        /*
            // 1. Interpolate horizontally along the top row between (i0, j0) and (i1, j0)
            auto top = lerp(AtByXYZ(i0, j0), AtByXYZ(i1, j0), u);

            // 2. Interpolate horizontally along the bottom row between (i0, j1) and (i1, j1)
            auto bottom = lerp(AtByXYZ(i0, j1), AtByXYZ(i1, j1), u);

            // 3. Finally, interpolate vertically between the top and bottom results using v
            return lerp(top, bottom, v);
        */
        return AtByXY(i0, j0) * ((1.0F - u) * (1.0F - v))
            + AtByXY(i1, j0) * (u * (1.0F - v))
            + AtByXY(i0, j1) * ((1.0F - u) * v)
            + AtByXY(i1, j1) * (u * v);
    }

    void texture::init(int32_t width, int32_t height, int32_t channels)
    {
        width_ = width;
        height_ = height;
        m_channels = channels;

        m_size = height * width;

        value_.clear();
        value_.resize(width * height);
    }

    void texture::Fill(const aten::vec4& value)
    {
        std::fill(value_.begin(), value_.end(), value);
    }

    bool texture::initAsGLTexture()
    {
        if (m_gltex == 0) {
            AT_VRETURN(width_ > 0, false);
            AT_VRETURN(height_ > 0, false);
            AT_VRETURN(value_.size() > 0, false);

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
                &value_[0]));

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
        AT_VRETURN(value_.size() == rhs.value_.size(), false);

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
        for (int32_t y = 0; y < height_; y++) {
            for (int32_t x = 0; x < width_; x++) {
                int32_t idx = y * width_ + x;

                value_[idx] += rhs.value_[idx];
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

                dst[yy * width_ + x].r() = (uint8_t)aten::clamp(value_[y * width_ + x].x * float(255), float(0), float(255));
                dst[yy * width_ + x].g() = (uint8_t)aten::clamp(value_[y * width_ + x].y * float(255), float(0), float(255));
                dst[yy * width_ + x].b() = (uint8_t)aten::clamp(value_[y * width_ + x].z * float(255), float(0), float(255));
            }
        }

        auto ret = ::stbi_write_png(filename.c_str(), width_, height_, bpp, &dst[0], pitch);
        AT_ASSERT(ret > 0);

        return (ret > 0);
    }

    void texture::SetFilterAndAddressModeAsGLTexture()
    {
        constexpr std::array GLFilterMode = {
            GL_NEAREST,
            GL_LINEAR,
        };
        AT_STATICASSERT(GLFilterMode.size() == static_cast<size_t>(TextureFilterMode::Max));

        constexpr std::array GLAddressMode = {
            GL_REPEAT,
            GL_CLAMP,
            GL_MIRRORED_REPEAT,
            GL_CLAMP_TO_BORDER,
        };
        AT_STATICASSERT(GLAddressMode.size() == static_cast<size_t>(TextureAddressMode::Max));

        const auto gl_filter = GLFilterMode[static_cast<int>(filter_mode_)];
        const auto gl_address = GLAddressMode[static_cast<int>(address_mode_)];

        // Specify filter after binding!!!!!
        CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, gl_address));
        CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, gl_address));
        CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, gl_filter));
        CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, gl_filter));
    }
}
