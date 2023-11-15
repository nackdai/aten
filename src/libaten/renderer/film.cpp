#include <string.h>
#include "renderer/film.h"

namespace aten
{
    Film::Film(int32_t w, int32_t h)
    {
        init(w, h);
    }

    void Film::init(int32_t w, int32_t h)
    {
        width_ = w;
        height_ = h;
        m_image.resize(width_ * height_);
    }

    void Film::clear()
    {
        memset(&m_image[0], 0, m_image.size() * sizeof(vec4));
    }

    void Film::put(int32_t x, int32_t y, const vec3& v)
    {
        put(x, y, vec4(v, 1));
    }

    void Film::put(int32_t i, const vec3& v)
    {
        put(i, vec4(v, 1));
    }

    void Film::put(int32_t x, int32_t y, const vec4& v)
    {
        x = aten::clamp<int32_t>(x, 0, width_ - 1);
        y = aten::clamp<int32_t>(y, 0, height_ - 1);

        auto pos = y * width_ + x;
        put(pos, v);
    }

    void Film::put(int32_t i, const vec4& v)
    {
        m_image[i] = v;
    }

    void Film::add(int32_t i, const vec4& v)
    {
        m_image[i] += v;
    }

    const vec4& Film::at(int32_t x, int32_t y) const
    {
        auto pos = y * width_ + x;
        return m_image[pos];
    }

    // NOTE
    // http://www.flint.jp/blog/?entry=86

    void FilmProgressive::put(int32_t i, const vec4& v)
    {
        auto& curValue = m_image[i];

        auto n = static_cast<real>(static_cast<int32_t>(curValue.w));

        curValue = n * curValue + v;
        curValue /= (n + 1);

        curValue.w = n + 1;
    }
}
