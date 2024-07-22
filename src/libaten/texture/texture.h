#pragma once

#include <memory>
#include <string>
#include <vector>

#include "defs.h"
#include "types.h"
#include "math/vec3.h"
#include "math/vec4.h"
#include "visualizer/shader.h"

namespace aten
{
    class texture {
        friend class context;

    public:
        texture() = default;
        texture(int32_t width, int32_t height, int32_t channels, std::string_view name);

        ~texture();

    private:
        static std::shared_ptr<texture> create(
            int32_t width, int32_t height, int32_t channels, std::string_view name);

    public:
        void init(int32_t width, int32_t height, int32_t channels);

        vec4 at(float u, float v) const
        {
            int32_t iu = static_cast<int32_t>(u * (width_ - 1));
            int32_t iv = static_cast<int32_t>(v * (height_ - 1));

            // NOTE:
            // Wrap as repeat.
            const auto x = NormalizeToWrapRepeat(iu, width_ - 1);
            const auto y = NormalizeToWrapRepeat(iv, height_ - 1);

            uint32_t pos = y * width_ + x;

            const auto clr = m_colors[pos];

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

        float& operator()(int32_t x, int32_t y, int32_t c)
        {
            x = std::min(x, width_ - 1);
            y = std::min(y, height_ - 1);
            c = std::min(c, m_channels - 1);

            auto pos = ((height_ - 1) - y) * width_ + x;

            return m_colors[pos][c];
        }

        const vec4* colors() const
        {
            return &m_colors[0];
        }

        auto width() const
        {
            return width_;
        }

        auto height() const
        {
            return height_;
        }

        auto channels() const
        {
            return m_channels;
        }

        auto id() const
        {
            return m_id;
        }

        const char* name() const
        {
            return m_name.c_str();
        }

        const std::string& nameString() const
        {
            return m_name;
        }

        bool initAsGLTexture();
        bool initAsGLTexture(int32_t width, int32_t height);
        void bindAsGLTexture(uint8_t stage, shader* shd) const;
        void releaseAsGLTexture();
        void clearAsGLTexture(const aten::vec4& clearColor);

        void getDataAsGLTexture(
            int32_t& width,
            int32_t& height,
            int32_t& channel,
            std::vector<vec4>& dst) const;

        static void bindAsGLTexture(
            uint32_t gltex,
            uint8_t stage, shader* shd);

        uint32_t getGLTexHandle() const
        {
            return m_gltex;
        }

        bool merge(const texture& rhs);

        bool exportAsPNG(const std::string& filename);

    private:
        template <class T>
        auto updateIndex(T id)
            -> std::enable_if_t<(std::is_signed<T>::value && !std::is_floating_point<T>::value) || std::is_same<T, std::size_t>::value, void>
        {
            m_id = static_cast<decltype(m_id)>(id);
        }

        static int32_t NormalizeToWrapRepeat(int32_t value, int32_t wrap_size)
        {
            if (value > wrap_size) {
                auto n = value / wrap_size;
                value -= n * wrap_size;
            }
            else if (value < 0) {
                auto n = aten::abs(value / wrap_size);
                value += (n + 1) * wrap_size;
            }
            return value;
        }

    private:
        int32_t m_id{ -1 };

        int32_t width_{ 0 };
        int32_t height_{ 0 };
        int32_t m_channels{ 0 };

        int32_t m_size{ 0 };

        std::vector<vec4> m_colors;

        uint32_t m_gltex{ 0 };

        std::string m_name;
    };
}
