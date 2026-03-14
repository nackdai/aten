#pragma once

#include <vector>

#include "defs.h"
#include "math/vec4.h"

namespace aten {
    class texture3d {
    public:
        texture3d() = default;
        texture3d(int32_t width, int32_t height, int32_t depth)
        {
            init(width, height, depth);
        }

        texture3d(int32_t width, int32_t height, int32_t depth, const aten::vec4& default)
        {
            init(width, height, depth);
            Fill(default);
        }

        ~texture3d() = default;

        void init(int32_t width, int32_t height, int32_t depth)
        {
            width_ = width;
            height_ = height;
            depth_ = depth;

            value_.clear();
            value_.resize(width * height * depth);
        }

        bool empty() const
        {
            return value_.empty();
        }

        void Fill(const aten::vec4& value)
        {
            std::fill(value_.begin(), value_.end(), value);
        }

        // Nearest neighbor sampling
        aten::vec4 at(float u, float v, float w) const
        {
            int32_t iu = static_cast<int32_t>(u * (width_ - 1));
            int32_t iv = static_cast<int32_t>(v * (height_ - 1));
            int32_t iw = static_cast<int32_t>(w * (depth_ - 1));

            const auto x = NormalizeToWrapRepeat(iu, width_ - 1);
            const auto y = NormalizeToWrapRepeat(iv, height_ - 1);
            const auto z = NormalizeToWrapRepeat(iw, depth_ - 1);

            return AtByXYZ(x, y, z);
        }

        vec4 at(const vec3& coord) const
        {
            return at(coord.x, coord.y, coord.z);
        }

        aten::vec4 AtWithTrilinear(float x, float y, float z) const
        {
            float u = x * width_ - 0.5F;
            float v = y * height_ - 0.5F;
            float w = z * depth_ - 0.5F;
            const int32_t i = static_cast<int32_t>(aten::floor(u));
            const int32_t j = static_cast<int32_t>(aten::floor(v));
            const int32_t k = static_cast<int32_t>(aten::floor(w));
            u -= i;
            v -= j;
            w -= k;

            int i0 = aten::max(0, aten::min(width_ - 1, i));
            int i1 = aten::max(0, aten::min(width_ - 1, i + 1));
            int j0 = aten::max(0, aten::min(height_ - 1, j));
            int j1 = aten::max(0, aten::min(height_ - 1, j + 1));
            int k0 = aten::max(0, aten::min(depth_ - 1, k));
            int k1 = aten::max(0, aten::min(depth_ - 1, k + 1));

            return AtByXYZ(i0, j0, k0) * ((1.0F - u) * (1.0F - v) * (1.0F - w)) +
                AtByXYZ(i1, j0, k0) * (u * (1.0F - v) * (1.0F - w)) +
                AtByXYZ(i0, j1, k0) * ((1.0F - u) * v * (1.0F - w)) +
                AtByXYZ(i1, j1, k0) * (u * v * (1.0F - w)) +
                AtByXYZ(i0, j0, k1) * ((1.0F - u) * (1.0F - v) * w) +
                AtByXYZ(i1, j0, k1) * (u * (1.F - v) * w) +
                AtByXYZ(i0, j1, k1) * ((1.0F - u) * v * w) +
                AtByXYZ(i1, j1, k1) * (u * v * w);
        }

        // Direct indexing
        virtual aten::vec4 AtByXYZ(int32_t x, int32_t y, int32_t z) const
        {
            uint32_t pos = (z * height_ + y) * width_ + x;
            return value_[pos];
        }

        void SetByUVW(
            const aten::vec4& value,
            float u, float v, float w)
        {
            int32_t iu = static_cast<int32_t>(u * (width_ - 1));
            int32_t iv = static_cast<int32_t>(v * (height_ - 1));
            int32_t iw = static_cast<int32_t>(w * (depth_ - 1));

            const auto x = NormalizeToWrapRepeat(iu, width_ - 1);
            const auto y = NormalizeToWrapRepeat(iv, height_ - 1);
            const auto z = NormalizeToWrapRepeat(iw, depth_ - 1);

            SetByXYZ(value, x, y, z);
        }

        void SetByXYZ(
            const aten::vec4& value,
            int32_t x, int32_t y, int32_t z)
        {
            uint32_t pos = (z * height_ + y) * width_ + x;
            value_[pos] = value;
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

        const std::vector<aten::vec4>& data() const
        {
            return value_;
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

    protected:
        int32_t width_{ 0 };
        int32_t height_{ 0 };
        int32_t depth_{ 0 };

        std::vector<aten::vec4> value_;
    };
}
