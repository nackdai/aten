#include "angularmap.h"

#include "atenscene.h"

std::tuple<float, float, CubemapFace> AngularMap::GetUVFromDir(const aten::vec3& dir) const
{
    float u = 0.0F;
    float v = 0.0F;

    if (aten::isClose(dir.z, 1.0F, 1000)) {
        // If the direction is (0, 0, 1), it means the direction is alomst the same as z-axist.
        // Then, return the center of the texture.
        u = 0.0f;
        v = 0.0f;
    }
    else {
        const auto alpha = aten::acos(dir.z);

        const auto radius = alpha / AT_MATH_PI;

        const auto sin = aten::sqrt(1.0F - dir.z * dir.z);

        // NOTE
        // dir.x = sin * u / r <-> u = dir.x * r / sin
        // dir.y = sin * v / r <-> v = dir.y * r / sin

        u = -dir.x * radius / (sin + 0.0001F);
        v = dir.y * radius / (sin + 0.0001F);

        u = aten::clamp(u, -1.0F, 1.0F);
        v = aten::clamp(v, -1.0F, 1.0F);

        // NOTE:
        // The direction of +x is -u direction. So, we need to flip u.

        u = (u + 1.0F) * 0.5F;
        v = (v + 1.0F) * 0.5F;
    }

    return std::make_tuple(u, v, CubemapFace::Num);
}

aten::vec3 AngularMap::GetDirFromXY(
    int32_t x, int32_t y,
    CubemapFace face/*= CubemapFace::Num*/) const
{
    std::ignore = face;

    const auto width = static_cast<float>(tex_->width() - 1);
    const auto height = static_cast<float>(tex_->height() - 1);

    const auto u = 2.0F * (x / width) - 1.0F;
    const auto v = 2.0F * (y / height) - 1.0F;

    const auto radius = aten::sqrt(u * u + v * v);

    aten::vec3 dir;

    if (radius < 0.0001F) {
        // We assume the center means the z-asis direction.
        dir.x = 0.0f;
        dir.y = 0.0f;
        dir.z = 1.0f;
    }
    else {
        auto alpha = radius * AT_MATH_PI;
        alpha = (u < 0.0f ? -alpha : alpha);

        dir.z = aten::cos(alpha);

        const auto sin = ::sqrtf(1.0F - dir.z * dir.z);

        // NOTE:
        // The direction of +x is -u direction. So, we need to flip u.

        dir.x = sin * -u / radius;
        dir.y = sin * v / radius;
    }

    dir = normalize(dir);

    return dir;
}

bool AngularMap::IsValidPos(int32_t x, int32_t y) const
{
    // We assume the width is equal with height;
    const auto width = this->width();
    const auto height = this->height();

    if (width != height) {
        AT_ASSERT(false);
        return false;
    }

    float u = x / static_cast<float>(width - 1);
    float v = y / static_cast<float>(height - 1);

    // [0, 1] -> [-1, 1]
    u = 2.0F * u - 1.0F;
    v = 2.0F * v - 1.0F;

    return ((u * u + v * v) <= 1.0F);
}
