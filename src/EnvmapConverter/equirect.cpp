#include "equirect.h"

#include "atenscene.h"

// NOTE
// -pi <= phi <= pi
// -pi/2 <= theta <= pi/2

// NOTE
// 0.0f <= u,v <= 1.0f

// NOTE
// x = cos(theta) * sin(phi)
// y = sin(theta)
// z = cos(theta) * cos(phi)

// NOTE
//          |-
//  +phi----+---- -phi
//          |+
//           theta

std::tuple<float, float, CubemapFace> EquirectMap::GetUVFromDir(const aten::vec3& dir) const
{
    const auto uv = aten::Background::ConvertDirectionToUV(dir);

    auto u = uv.x;
    if (u < 0.5F) {
        u = 0.5F - u;
    }
    else {
        u = 1.0F - (u - 0.5F);
    }

    const auto v = uv.y;

    return std::make_tuple(u, v, CubemapFace::Num);
}

aten::vec3 EquirectMap::GetDirFromXY(
    int32_t x, int32_t y,
    CubemapFace face/*= CubemapFace::Num*/) const
{
    std::ignore = face;

    const auto width = static_cast<float>(tex_->width() - 1);
    const auto height = static_cast<float>(tex_->height() - 1);

    float phi = 0.0F;

    auto u = x / width;
    if (u < 0.5F) {
        u = 0.5F - u;
        phi = u * AT_MATH_PI_2;
    }
    else {
        u = 1.0F - (u - 0.5F);
        phi = u * AT_MATH_PI_2 - AT_MATH_PI_2;
    }

    const auto v = 1.0F - y / height;
    const auto theta = AT_MATH_PI * v;

    const auto sin_theta = aten::sin(theta);
    const auto cos_theta = aten::cos(theta);
    const auto sin_phi = aten::sin(phi);
    const auto cos_phi = aten::cos(phi);

    aten::vec3 dir;
    dir.x = sin_theta * sin_phi;
    dir.y = cos_theta;
    dir.z = sin_theta * cos_phi;

    dir = normalize(dir);

    return dir;
}

aten::vec4 EquirectMap::At(
    float u, float v,
    CubemapFace face/*= CubemapFace::Num*/) const
{
    std::ignore = face;

    const auto color = tex_->at(u, v);
    return color;
}

void EquirectMap::Put(
    const aten::vec4& color,
    std::int32_t x, std::int32_t y,
    CubemapFace face/*= CubemapFace::Num*/)
{
    AT_ASSERT(tex_);

    if (tex_) {
        const auto width = static_cast<float>(tex_->width());
        const auto height = static_cast<float>(tex_->height());
        const auto u = x / width;
        const auto v = y / height;
        tex_->put(color, u, v);
    }
}

bool EquirectMap::SaveAsPng(std::string_view filename) const
{
    AT_ASSERT(tex_);

    bool result = false;

    if (tex_) {
        result = tex_->exportAsPNG(filename.data());
    }

    return result;
}
