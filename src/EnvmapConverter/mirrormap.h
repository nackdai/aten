#pragma once

#include "envmap.h"

class MirrorMap : public EnvMap {
public:
    MirrorMap() : EnvMap(EnvMapType::Equirect) {}
    virtual ~MirrorMap() = default;

    MirrorMap(const MirrorMap&) = delete;
    MirrorMap(MirrorMap&&) = delete;
    MirrorMap& operator=(const MirrorMap&) = delete;
    MirrorMap& operator=(MirrorMap&&) = delete;

    static MirrorMap* Load(
        aten::context& ctxt,
        std::string_view filename);

    static MirrorMap* Create(std::int32_t width, std::int32_t height);

    std::tuple<float, float, CubemapFace> GetUVFromDir(const aten::vec3& dir) const override final;

    aten::vec3 GetDirFromXY(
        int32_t x, int32_t y,
        CubemapFace face = CubemapFace::Num) const override final;

    aten::vec4 At(
        float u, float v,
        CubemapFace face = CubemapFace::Num) const override final;

    void Put(
        const aten::vec4& color,
        std::int32_t x, std::int32_t y,
        CubemapFace face = CubemapFace::Num) override final;

    bool SaveAsPng(std::string_view filename) const override final;

    int32_t width() const override final
    {
        AT_ASSERT(tex_);
        if (!tex_) {
            return 0;
        }
        return tex_->width();
    }

    int32_t height() const override final
    {
        AT_ASSERT(tex_);
        if (!tex_) {
            return 0;
        }
        return tex_->height();
    }

    bool IsValidPos(int32_t x, int32_t y) const override final;

protected:
    std::shared_ptr<aten::texture> tex_;
};

