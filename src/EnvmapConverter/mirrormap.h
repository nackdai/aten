#pragma once

#include "envmap.h"

class MirrorMap : public SingleEnvMap {
public:
    MirrorMap() : SingleEnvMap(EnvMapType::Mirror) {}
    virtual ~MirrorMap() = default;

    MirrorMap(const MirrorMap&) = delete;
    MirrorMap(MirrorMap&&) = delete;
    MirrorMap& operator=(const MirrorMap&) = delete;
    MirrorMap& operator=(MirrorMap&&) = delete;

    std::tuple<float, float, CubemapFace> GetUVFromDir(const aten::vec3& dir) const override final;

    aten::vec3 GetDirFromXY(
        int32_t x, int32_t y,
        CubemapFace face = CubemapFace::Num) const override final;

    bool IsValidPos(int32_t x, int32_t y) const override final;
};
