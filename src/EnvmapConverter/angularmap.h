#pragma once

#include "envmap.h"

class AngularMap : public SingleEnvMap {
public:
    AngularMap() : SingleEnvMap(EnvMapType::Equirect) {}
    virtual ~AngularMap() = default;

    AngularMap(const AngularMap&) = delete;
    AngularMap(AngularMap&&) = delete;
    AngularMap& operator=(const AngularMap&) = delete;
    AngularMap& operator=(AngularMap&&) = delete;

    std::tuple<float, float, CubemapFace> GetUVFromDir(const aten::vec3& dir) const override final;

    aten::vec3 GetDirFromXY(
        int32_t x, int32_t y,
        CubemapFace face = CubemapFace::Num) const override final;

    bool IsValidPos(int32_t x, int32_t y) const override final;
};
