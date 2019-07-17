#pragma once

#include <vector>
#include "types.h"
#include "math/vec4.h"

namespace aten
{
    class Film {
    public:
        Film() {}
        Film(int w, int h);
        virtual ~Film() {}

    public:
        void init(int w, int h);

        virtual void clear();

        void put(int x, int y, const vec3& v);
        void put(int i, const vec3& v);

        void put(int x, int y, const vec4& v);
        virtual void put(int i, const vec4& v);

        virtual void add(int i, const vec4& v);

        const vec4& at(int x, int y) const;

        vec4* image()
        {
            if (m_image.size() > 0) {
                return &m_image[0];
            }
            return nullptr;
        }

        const vec4* image() const
        {
            if (m_image.size() > 0) {
                return &m_image[0];
            }
            return nullptr;
        }

        uint32_t width() const
        {
            return m_width;
        }

        uint32_t height() const
        {
            return m_height;
        }

    protected:
        std::vector<vec4> m_image;
        int m_width{ 0 };
        int m_height{ 0 };
    };

    class FilmProgressive : public Film {
    public:
        FilmProgressive() {}
        FilmProgressive(int w, int h)
            : Film(w, h)
        {}
        virtual ~FilmProgressive() {}

    public:
        virtual void clear() final
        {
            // Nothing is done...
        }

        virtual void put(int i, const vec4& v) override final;

        virtual void add(int i, const vec4& v) override final
        {
            put(i, v);
        }
    };
}
