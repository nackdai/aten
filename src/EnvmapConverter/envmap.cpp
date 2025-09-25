#include "envmap.h"

#include "cubemap.h"
#include "equirect.h"
#include "mirrormap.h"

#include <memory>

std::shared_ptr<EnvMap> EnvMap::LoadEnvmap(
    aten::context& ctxt,
    EnvMapType type,
    std::string_view filename,
    std::string_view filename_neg_x/*= ""*/,
    std::string_view filename_pos_y/*= ""*/,
    std::string_view filename_neg_y/*= ""*/,
    std::string_view filename_pos_z/*= ""*/,
    std::string_view filename_neg_z/*= ""*/)
{
    EnvMap* envmap = nullptr;

    switch(type) {
    case EnvMapType::CubeMap:
        envmap = CubeMap::Load(
            ctxt,
            filename, filename_neg_x,
            filename_pos_y, filename_neg_y,
            filename_pos_z, filename_neg_z);
        break;
    case EnvMapType::Equirect:
        envmap = EquirectMap::Load(ctxt, filename);
        break;
    case EnvMapType::Mirror:
        envmap = MirrorMap::Load(ctxt, filename);
        break;
    }

    std::shared_ptr<EnvMap> ret(envmap);
    return ret;
}

std::shared_ptr<EnvMap> EnvMap::CreateEmptyEnvmap(
    EnvMapType type,
    std::int32_t width, std::int32_t height)
{
    EnvMap* envmap = nullptr;

    switch (type) {
    case EnvMapType::CubeMap:
        envmap = CubeMap::Create(width, height);
        break;
    case EnvMapType::Equirect:
        envmap = EquirectMap::Create(width, height);
        break;
    case EnvMapType::Mirror:
        envmap = MirrorMap::Create(width, height);
        break;
    }

    std::shared_ptr<EnvMap> ret(envmap);
    return ret;
}

void EnvMap::Convert(
    const std::shared_ptr<EnvMap>& src,
    const std::shared_ptr<EnvMap>& dst)
{
    const auto dst_width = dst->width();
    const auto dst_height = dst->height();

    const auto dst_type = dst->type();

    const int32_t dst_face_num = dst_type == EnvMapType::CubeMap
        ? static_cast<int32_t>(CubemapFace::Num)
        : 1;

    for (int32_t face = 0; face < dst_face_num; face++) {
        for (int32_t y = 0; y < dst_height; y++) {
            for (int32_t x = 0; x < dst_width; x++) {
                if (!dst->IsValidPos(x, y)) {
                    continue;
                }

                const auto dir = dst->GetDirFromXY(x, y, static_cast<CubemapFace>(face));
                const auto [u, v, target_face] = src->GetUVFromDir(dir);

                const auto color = src->At(u, v, target_face);
                dst->Put(color, x, y, static_cast<CubemapFace>(face));
            }
        }
    }
}
