#pragma once

#include "envmap.h"

class CubeMap : public EnvMap {
public:
    CubeMap() : EnvMap(EnvMapType::CubeMap) {}
    virtual ~CubeMap() = default;

    CubeMap(const CubeMap&) = delete;
    CubeMap(CubeMap&&) = delete;
    CubeMap& operator=(const CubeMap&) = delete;
    CubeMap& operator=(CubeMap&&) = delete;

    static CubeMap* Load(
        aten::context& ctxt,
        std::string_view filename_pos_x,
        std::string_view filename_neg_x,
        std::string_view filename_pos_y,
        std::string_view filename_neg_y,
        std::string_view filename_pos_z,
        std::string_view filename_neg_z);

    static CubeMap* Create(std::int32_t width, std::int32_t height);

    std::tuple<float, float, CubemapFace> GetUVFromDir(const aten::vec3 & dir) const override final;

    aten::vec3 GetDirFromXY(
        int32_t x, int32_t y,
        CubemapFace face = CubemapFace::Num) const override final;

    aten::vec4 At(
        float u, float v,
        CubemapFace face = CubemapFace::Num) const override final;

    void Put(
        const aten::vec4 & color,
        std::int32_t x, std::int32_t y,
        CubemapFace face = CubemapFace::Num) override final;

    bool SaveAsPng(std::string_view filename) const override final;

    int32_t width() const override final
    {
        AT_ASSERT(!faces_.empty() && faces_[0]);
        if (faces_.empty() || !faces_[0]) {
            return 0;
        }
        return faces_[0]->width();
    }

    int32_t height() const override final
    {
        AT_ASSERT(!faces_.empty() && faces_[0]);
        if (faces_.empty() || !faces_[0]) {
            return 0;
        }
        return faces_[0]->height();
    }

protected:
    std::vector<std::shared_ptr<aten::texture>> faces_;
};
