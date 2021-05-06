#include <string.h>
#include "renderer/film.h"

namespace aten
{
    Film::Film(int w, int h)
    {
        init(w, h);
    }

    void Film::init(int w, int h)
    {
        m_width = w;
        m_height = h;
        m_image.resize(m_width * m_height);
    }

    void Film::clear()
    {
        memset(&m_image[0], 0, m_image.size() * sizeof(vec4));
    }

    void Film::put(int x, int y, const vec3& v)
    {
        put(x, y, vec4(v, 1));
    }

    void Film::put(int i, const vec3& v)
    {
        put(i, vec4(v, 1));
    }

    void Film::put(int x, int y, const vec4& v)
    {
        x = aten::clamp<int>(x, 0, m_width - 1);
        y = aten::clamp<int>(y, 0, m_height - 1);

        auto pos = y * m_width + x;
        put(pos, v);
    }

    void Film::put(int i, const vec4& v)
    {
        m_image[i] = v;
    }

    void Film::add(int i, const vec4& v)
    {
        m_image[i] += v;
    }

    const vec4& Film::at(int x, int y) const
    {
        auto pos = y * m_width + x;
        return m_image[pos];
    }

    // NOTE
    // http://www.flint.jp/blog/?entry=86

    void FilmProgressive::put(int i, const vec4& v)
    {
        auto& curValue = m_image[i];

        auto n = static_cast<real>(static_cast<int>(curValue.w));

        curValue = n * curValue + v;
        curValue /= (n + 1);

        curValue.w = n + 1;
    }
}
