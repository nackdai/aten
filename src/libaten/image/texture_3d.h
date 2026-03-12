#pragma once

#include <vector>

#include "defs.h"

namespace aten {
    template <class T>
    class texture3d {
    public:
        using ValueType = T;

        texture3d() = default;
        texture3d(int32_t width, int32_t height, int32_t depth)
        {
            init(width, height, depth);
        }

        ~texture3d() = default;

        void init(int32_t width, int32_t height, int32_t depth)
        {
            width_ = width;
            height_ = height;
            depth_ = depth;

            data_.clear();
            data_.resize(width * height * depth);
        }

        bool empty() const
        {
            return data_.empty();
        }

        void Fill(const ValueType& value)
        {
            std::fill(data_.begin(), data_.end(), value);
        }

        // Nearest neighbor sampling
        ValueType at(float u, float v, float w) const
        {
            int32_t iu = static_cast<int32_t>(u * (width_ - 1));
            int32_t iv = static_cast<int32_t>(v * (height_ - 1));
            int32_t iw = static_cast<int32_t>(w * (depth_ - 1));

            const auto x = NormalizeToWrapRepeat(iu, width_ - 1);
            const auto y = NormalizeToWrapRepeat(iv, height_ - 1);
            const auto z = NormalizeToWrapRepeat(iw, depth_ - 1);

            return AtByXYZ(x, y, z);
        }

        vec4 at(const vec3& coord) const override final
        {
            return at(coord.x, coord.y, coord.z);
        }

        ValueType AtWithTrilinear(float u, float v, float w) const
        {
            // Trilinear interpolation
            // https://en.wikipedia.org/wiki/Trilinear_interpolation

            const float fx = u * (width_ - 1);
            const float fy = v * (height_ - 1);
            const float fz = w * (depth_ - 1);

            float frac_x = fx - 0.5F - static_cast<int32_t>(fx);
            float frac_y = fy - 0.5F - static_cast<int32_t>(fy);
            float frac_z = fz - 0.5F - static_cast<int32_t>(fz);

            const auto x = static_cast<int32_t>(fx);
            const auto y = static_cast<int32_t>(fy);
            const auto z = static_cast<int32_t>(fz);

            auto nearest_x = x;
            if (frac_x >= 0.5F) {
                nearest_x = x + 1;
            }
            else {
                nearest_x = x - 1;
                frac_x = 1.0F - frac_x;
            }

            auto nearest_y = y;
            if (frac_y >= 0.5F) {
                nearest_y = y + 1;
            }
            else {
                nearest_y = y - 1;
                frac_y = 1.0F - frac_y;
            }

            auto nearest_z = z;
            if (frac_z >= 0.5F) {
                nearest_z = z + 1;
            }
            else {
                nearest_z = z - 1;
                frac_z = 1.0F - frac_z;
            }

            nearest_x = aten::clamp(nearest_x, 0, width_ - 1);
            nearest_y = aten::clamp(nearest_y, 0, height_ - 1);
            nearest_z = aten::clamp(nearest_z, 0, depth_ - 1);

            // Get sample values of 8 corners
            const auto c000 = AtByXYZ(x, y, z);
            const auto c100 = AtByXYZ(nearest_x, y, z);
            const auto c010 = AtByXYZ(x, nearest_y, z);
            const auto c110 = AtByXYZ(nearest_x, nearest_y, z);
            const auto c001 = AtByXYZ(x, y, nearest_z);
            const auto c101 = AtByXYZ(nearest_x, y, nearest_z);
            const auto c011 = AtByXYZ(x, nearest_y, nearest_z);
            const auto c111 = AtByXYZ(nearest_x, nearest_y, nearest_z);

            // XY plane interpolation (z=0 plane)
            const auto c00 = aten::lerp(c000, c100, frac_x);
            const auto c10 = aten::lerp(c010, c110, frac_x);
            const auto c0 = aten::lerp(c00, c10, frac_y);

            // XY plane interpolation (z=1 plane)
            const auto c01 = aten::lerp(c001, c101, frac_x);
            const auto c11 = aten::lerp(c011, c111, frac_x);
            const auto c1 = aten::lerp(c01, c11, frac_y);

            // Z direction interpolation
            const auto c = aten::lerp(c0, c1, frac_z);

            return c;
        }

        // Direct indexing
        ValueType AtByXYZ(int32_t x, int32_t y, int32_t z) const
        {
            uint32_t pos = (z * height_ + y) * width_ + x;
            return data_[pos];
        }

        void put(
            const ValueType& color,
            float u, float v, float w)
        {
            int32_t iu = static_cast<int32_t>(u * (width_ - 1));
            int32_t iv = static_cast<int32_t>(v * (height_ - 1));
            int32_t iw = static_cast<int32_t>(w * (depth_ - 1));

            const auto x = NormalizeToWrapRepeat(iu, width_ - 1);
            const auto y = NormalizeToWrapRepeat(iv, height_ - 1);
            const auto z = NormalizeToWrapRepeat(iw, depth_ - 1);

            uint32_t pos = (z * height_ + y) * width_ + x;
            data_[pos] = color;
        }

        auto width() const
        {
            return width_;
        }

        auto height() const
        {
            return height_;
        }

        auto depth() const
        {
            return depth_;
        }

        const std::vector<ValueType>& data() const
        {
            return data_;
        }

    private:
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
        int32_t width_{ 0 };
        int32_t height_{ 0 };
        int32_t depth_{ 0 };

        std::vector<ValueType> data_;
    };
}
