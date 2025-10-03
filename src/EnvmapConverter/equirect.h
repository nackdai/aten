#pragma once

#include "envmap.h"

class EquirectMap : public SingleEnvMap {
public:
    EquirectMap() : SingleEnvMap(EnvMapType::Equirect) {}
    virtual ~EquirectMap() = default;

    EquirectMap(const EquirectMap&) = delete;
    EquirectMap(EquirectMap&&) = delete;
    EquirectMap& operator=(const EquirectMap&) = delete;
    EquirectMap& operator=(EquirectMap&&) = delete;

    static EquirectMap* Load(
        aten::context& ctxt,
        std::string_view filename);

    static EquirectMap* Create(std::int32_t width, std::int32_t height);

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
};
