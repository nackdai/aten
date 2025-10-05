#include <memory>

#include "envmap.h"

#include "angularmap.h"
#include "cubemap.h"
#include "equirect.h"
#include "mirrormap.h"

#include "atenscene.h"

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
        envmap = SingleEnvMap::Load<EquirectMap>(ctxt, filename);
        break;
    case EnvMapType::Mirror:
        envmap = SingleEnvMap::Load<MirrorMap>(ctxt, filename);
        break;
    case EnvMapType::Angular:
        envmap = SingleEnvMap::Load<AngularMap>(ctxt, filename);
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
        envmap = SingleEnvMap::Create<EquirectMap>(width, height);
        break;
    case EnvMapType::Mirror:
        envmap = SingleEnvMap::Create<MirrorMap>(width, height);
        break;
    case EnvMapType::Angular:
        envmap = SingleEnvMap::Create<AngularMap>(width, height);
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

                const auto color = src->GetColorFromDir(dir);
                dst->Put(color, x, y, static_cast<CubemapFace>(face));
            }
        }
    }
}

///////////////////////////////////////////////

template<typename TEnvMap>
TEnvMap* SingleEnvMap::Load(
    aten::context& ctxt,
    std::string_view filename)
{
    auto tex = aten::ImageLoader::load(filename.data(), ctxt);
    if (!tex) {
        return nullptr;
    }

    auto envmap = new TEnvMap();
    if (!envmap) {
        return nullptr;
    }

    envmap->tex_ = std::move(tex);

    return envmap;
}

template<typename TEnvMap>
TEnvMap* SingleEnvMap::Create(std::int32_t width, std::int32_t height)
{
    auto envmap = new TEnvMap();
    if (!envmap) {
        return nullptr;
    }

    envmap->tex_ = std::make_shared<aten::texture>(width, height, 4, "");
    if (!envmap->tex_) {
        return nullptr;
    }

    return envmap;
}

aten::vec4 SingleEnvMap::At(
    float u, float v,
    CubemapFace face/*= CubemapFace::Num*/) const
{
    std::ignore = face;

    //const auto color = tex_->at(u, v);
    const auto color = tex_->AtWithBilinear(u, v);
    return color;
}

void SingleEnvMap::Put(
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

bool SingleEnvMap::SaveAsPng(std::string_view filename) const
{
    AT_ASSERT(tex_);

    bool result = false;

    if (tex_) {
        result = tex_->exportAsPNG(filename.data());
    }

    return result;
}
