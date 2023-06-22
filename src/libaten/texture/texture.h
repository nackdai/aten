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
        texture(int32_t width, int32_t height, uint32_t channels, std::string_view name);

        ~texture();

    private:
        static std::shared_ptr<texture> create(
            int32_t width, int32_t height, uint32_t channels, std::string_view name);

    public:
        void init(int32_t width, int32_t height, uint32_t channels);

        vec4 at(real u, real v) const
        {
            u -= floor(u);
            v -= floor(v);

            uint32_t x = (uint32_t)(aten::cmpMin(u, real(1)) * (m_width - 1));
            uint32_t y = (uint32_t)(aten::cmpMin(v, real(1)) * (m_height - 1));

            uint32_t pos = y * m_width + x;

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

        real& operator()(uint32_t x, uint32_t y, uint32_t c)
        {
            x = std::min(x, m_width - 1);
            y = std::min(y, m_height - 1);
            c = std::min(c, m_channels - 1);

            uint32_t pos = ((m_height - 1) - y) * m_width + x;

            return m_colors[pos][c];
        }

        const vec4* colors() const
        {
            return &m_colors[0];
        }

        uint32_t width() const
        {
            return m_width;
        }

        uint32_t height() const
        {
            return m_height;
        }

        uint32_t channels() const
        {
            return m_channels;
        }

        int32_t id() const
        {
            return m_id;
        }

        const char* name() const
        {
            return m_name.c_str();
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
        template <typename T>
        auto updateIndex(T id)
            -> std::enable_if_t<(std::is_signed<T>::value && !std::is_floating_point<T>::value) || std::is_same<T, std::size_t>::value, void>
        {
            m_id = static_cast<decltype(m_id)>(id);
        }

    private:
        int32_t m_id{ -1 };

        uint32_t m_width{ 0 };
        uint32_t m_height{ 0 };
        uint32_t m_channels{ 0 };

        uint32_t m_size{ 0 };

        std::vector<vec4> m_colors;

        uint32_t m_gltex{ 0 };

        std::string m_name;
    };
}
