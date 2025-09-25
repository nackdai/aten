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

EquirectMap* EquirectMap::Load(
    aten::context& ctxt,
    std::string_view filename)
{
    auto tex = aten::ImageLoader::load(filename.data(), ctxt);
    if (!tex) {
        return nullptr;
    }

    auto equirect = new EquirectMap();
    if (!equirect) {
        return nullptr;
    }

    equirect->tex_ = std::move(tex);

    return equirect;
}

EquirectMap* EquirectMap::Create(std::int32_t width, std::int32_t height)
{
    auto equirect = new EquirectMap();
    if (!equirect) {
        return nullptr;
    }

    equirect->tex_ = std::make_shared<aten::texture>(width, height, 4, "");
    if (!equirect->tex_) {
        return nullptr;
    }

    return equirect;
}

std::tuple<float, float, CubemapFace> EquirectMap::GetUVFromDir(const aten::vec3& dir) const
{
    const auto uv = aten::Background::ConvertDirectionToUV(dir);
    const auto u = uv.x;
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

    // NOTE:
    //          |-
    //  +phi----+---- -phi
    //          |+
    //           theta

    // [0,1] -> [-1,1]
    auto phi = 2.0f * (x / width) - 1.0f;
    auto theta = 2.0f * (y / height) - 1.0f;

    // NOTE:
    // -----+-----> phi => +phi----+---- -phi
    phi = -phi;

    phi = phi * AT_MATH_PI;
    theta = theta * AT_MATH_PI_HALF;

    const auto sin_theta = aten::sin(theta);
    const auto cos_theta = aten::cos(theta);
    const auto sin_phi = aten::sin(phi);
    const auto cos_phi = aten::cos(phi);

    aten::vec3 dir;
    dir.x = cos_theta * sin_phi;
    dir.y = sin_theta;
    dir.z = cos_theta * cos_phi;

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
