#pragma once

#include <vector>
#include "types.h"
#include "math/vec4.h"

namespace aten
{
    class Film {
    public:
        Film() {}
        Film(int32_t w, int32_t h);
        virtual ~Film() {}

    public:
        void init(int32_t w, int32_t h);

        virtual void clear();

        void put(int32_t x, int32_t y, const vec3& v);
        void put(int32_t i, const vec3& v);

        void put(int32_t x, int32_t y, const vec4& v);
        virtual void put(int32_t i, const vec4& v);

        virtual void add(int32_t i, const vec4& v);

        const vec4& at(int32_t x, int32_t y) const;

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
            return width_;
        }

        uint32_t height() const
        {
            return height_;
        }

    protected:
        std::vector<vec4> m_image;
        int32_t width_{ 0 };
        int32_t height_{ 0 };
    };

    class FilmProgressive : public Film {
    public:
        FilmProgressive() {}
        FilmProgressive(int32_t w, int32_t h)
            : Film(w, h)
        {}
        virtual ~FilmProgressive() {}

    public:
        virtual void clear() final
        {
            // Nothing is done...
        }

        virtual void put(int32_t i, const vec4& v) override final;

        virtual void add(int32_t i, const vec4& v) override final
        {
            put(i, v);
        }
    };
}
