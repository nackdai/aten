#include "cubemap.h"

#include "atenscene.h"

namespace _cubmap_detail {
    constexpr std::array face_type_str = {
        "pos_x", "neg_x",
        "pos_y", "neg_y",
        "pos_z", "neg_z",
    };
}

CubeMap* CubeMap::Load(
    aten::context& ctxt,
    std::string_view filename_pos_x,
    std::string_view filename_neg_x,
    std::string_view filename_pos_y,
    std::string_view filename_neg_y,
    std::string_view filename_pos_z,
    std::string_view filename_neg_z)
{
    auto cubemap = new CubeMap();
    if (!cubemap) {
        return nullptr;
    }

    const std::array filenames = {
        filename_pos_x.data(),
        filename_neg_x.data(),
        filename_pos_y.data(),
        filename_neg_y.data(),
        filename_pos_z.data(),
        filename_neg_z.data()
    };

    bool need_create_path = filename_neg_x.empty()
        || filename_pos_y.empty()
        || filename_neg_y.empty()
        || filename_pos_z.empty()
        || filename_neg_z.empty();

    for (int32_t face = 0; face < static_cast<int32_t>(CubemapFace::Num); face++) {
        const auto* filename = filenames[face];
        std::string face_png_name;

        if (need_create_path) {
            filename = filenames[0];

            std::string path;
            std::string file;
            std::string ext;

            aten::getStringsFromPath(
                filename,
                path, ext, file);

            face_png_name = path + file + "_" + _cubmap_detail::face_type_str[face] + ext;
        }
        else {
            face_png_name = filenames[face];
        }

        auto tex = aten::ImageLoader::load(face_png_name, ctxt);
        if (!tex) {
            delete cubemap;
            return nullptr;
        }

        cubemap->faces_.emplace_back(std::move(tex));
    }

    return cubemap;
}

CubeMap* CubeMap::Create(std::int32_t width, std::int32_t height)
{
    auto cubemap = new CubeMap();
    if (!cubemap) {
        return nullptr;
    }

    constexpr auto face_num = static_cast<int>(CubemapFace::Num);

    for (int i = 0; i < face_num; i++) {
        auto tex = std::make_shared<aten::texture>(width, height, 4, "");
        if (!tex) {
            delete cubemap;
            return nullptr;
        }

        cubemap->faces_.emplace_back(std::move(tex));
    }

    return cubemap;
}

std::tuple<float, float, CubemapFace> CubeMap::GetUVFromDir(const aten::vec3& dir) const
{
    /**
        o--------+--->X
        |\       |
        |  \     |
        |    \   |
        |      \ |
        +--------p
        |         v(x, z)
        +Z

        cube mapは 90度ごとに面が変わるので、ちょうどxの端までのベクトルv (o->p)のなす角度は45度になる
        つまり、z / x = tan(theta) = tan(45度) = 1.0 で最大値になるので、この場合は、z / x をすることで u 座標値が計算できる。

        Cube map changes its face per 90 degree. In that case, the angle (= theta) of vector v (= o -> p) at o is 45 degree.
        Then, z / x = tan(theta) = tan(45 degree) = 1.0 and it is max of z / x.
        Therefore, in this case, we can computes the u coordinate value of the positive X face as z / x.
    **/

    // Decide which face should be picked from the max abs value of direction vector.
    const auto x = aten::abs(dir.x);
    const auto y = aten::abs(dir.y);
    const auto z = aten::abs(dir.z);

    const auto max_dir = aten::max(x, aten::max(y, z));

    CubemapFace face;

    if (max_dir == x) {
        // X
        face = (dir.x > 0.0f ? CubemapFace::PosX : CubemapFace::NegX);
    }
    else if (max_dir == y) {
        // Y
        face = (dir.y > 0.0f ? CubemapFace::PosY : CubemapFace::NegY);
    }
    else {
        // Z
        face = (dir.z > 0.0f ? CubemapFace::PosZ : CubemapFace::NegZ);
    }

    float u = 0.0f;
    float v = 0.0f;

    // Compute uv coordinate.
    switch (face) {
    case CubemapFace::PosX: // +X
        u = dir.z / dir.x;
        v = dir.y / dir.x;
        break;
    case CubemapFace::NegX: // -X
        u = dir.z / dir.x;
        v = -dir.y / dir.x;
        break;
    case CubemapFace::PosY: // +Y
        u = dir.x / dir.y;
        v = dir.z / dir.y;
        break;
    case CubemapFace::NegY: // -Y
        u = dir.x / dir.y;
        v = -dir.z / dir.y;
        break;
    case CubemapFace::PosZ: // +Z
        u = -dir.x / dir.z;
        v = dir.y / dir.z;
        break;
    case CubemapFace::NegZ: // -Z
        u = -dir.x / dir.z;
        v = -dir.y / dir.z;
        break;
    }

    u = aten::clamp(u, -1.0f, 1.0f);
    v = aten::clamp(v, -1.0f, 1.0f);

    // Convert [-1,1] -> [0,1]
    u = (u + 1.0f) * 0.5f;
    v = (v + 1.0f) * 0.5f;

    return std::make_tuple(u, v, face);
}

aten::vec3 CubeMap::GetDirFromXY(
    int32_t x, int32_t y,
    CubemapFace face/*= CubemapFace::Num*/) const
{
    AT_ASSERT(face < CubemapFace::Num);

    const auto face_tex = faces_[static_cast<int>(face)];

    const auto width = static_cast<float>(face_tex->width() - 1);
    const auto height = static_cast<float>(face_tex->height() - 1);

    // Conver xy to uv as [-1,1].
    const auto u = 2.0f * (x / width) - 1.0f;
    const auto v = 2.0f * (y / height) - 1.0f;

    // NOTE:
    // If we see each face in front of the face, we need to consider how u,v axis are aligned.
    // 各面を正面から見たときにu、v方向がどの軸になるのか
    //
    // ex) +X face case.
    //       +v
    //    +---|---+
    // +u |   |   |
    // <--+---+---+-
    //    |   |   |
    //    +---|---+

    aten::vec3 dir;

    switch (face) {
    case CubemapFace::PosX: // +X
        dir.x = 1.0f;
        dir.y = v;
        dir.z = u;
        break;
    case CubemapFace::NegX: // -X
        dir.x = -1.0f;
        dir.y = v;
        dir.z = -u;
        break;
    case CubemapFace::PosY: // +Y
        dir.x = -u;
        dir.y = 1.0f;
        dir.z = -v;
        break;
    case CubemapFace::NegY: // -Y
        dir.x = -u;
        dir.y = -1.0f;
        dir.z = v;
        break;
    case CubemapFace::PosZ: // +Z
        dir.x = -u;
        dir.y = v;
        dir.z = 1.0f;
        break;
    case CubemapFace::NegZ: // -Z
        dir.x = u;
        dir.y = v;
        dir.z = -1.0f;
        break;
    }

    dir = normalize(dir);
    return dir;
}

aten::vec4 CubeMap::At(
    float u, float v,
    CubemapFace face/*= CubemapFace::Num*/) const
{
    AT_ASSERT(face < CubemapFace::Num);

    const auto face_tex = faces_[static_cast<int>(face)];

    const auto color = face_tex->at(u, v);
    return color;
}

void CubeMap::Put(
    const aten::vec4& color,
    std::int32_t x, std::int32_t y,
    CubemapFace face/*= CubemapFace::Num*/)
{
    AT_ASSERT(face < CubemapFace::Num);

    const auto face_tex = faces_[static_cast<int>(face)];
    AT_ASSERT(face_tex);

    if (face_tex) {
        const auto width = static_cast<float>(face_tex->width());
        const auto height = static_cast<float>(face_tex->height());
        const auto u = x / width;
        const auto v = y / height;
        face_tex->put(color, u, v);
    }
}

bool CubeMap::SaveAsPng(std::string_view filename) const
{
    std::string path;
    std::string file;
    std::string ext;

    aten::getStringsFromPath(
        filename,
        path, ext, file);

    constexpr auto face_num = static_cast<int32_t>(CubemapFace::Num);

    bool result = true;

    for (int32_t face = 0; face < face_num; face++) {
        std::string face_png_name = path + file + "_" + _cubmap_detail::face_type_str[face] + ext;

        const auto& face_tex_ = faces_[face];
        AT_ASSERT(face_tex_);

        result |= (face_tex_ != nullptr);

        if (result) {
            result |= face_tex_->exportAsPNG(face_png_name);
        }

        AT_ASSERT(result);
    }

    return result;
}
