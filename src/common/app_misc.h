#pragma once

#include <memory>

#include "aten.h"

inline std::shared_ptr<aten::material> CreateMaterial(
    std::string_view name,
    aten::context& ctxt,
    aten::MaterialType type,
    const aten::vec3& albedo,
    aten::texture* albedoMap,
    aten::texture* normalMap)
{
    aten::MaterialParameter param;
    param.type = type;
    param.baseColor = albedo;

    auto mtrl = ctxt.CreateMaterialWithMaterialParameter(
        name,
        param,
        albedoMap,
        normalMap,
        nullptr);

    return mtrl;
}

inline std::shared_ptr<aten::material> CreateMaterial(
    std::string_view name,
    aten::context& ctxt,
    aten::MaterialType type,
    const aten::vec3& albedo)
{
    aten::MaterialParameter param;
    param.type = type;
    param.baseColor = albedo;

    return ctxt.CreateMaterialWithMaterialParameter(
        name,
        param,
        nullptr, nullptr, nullptr);
}

inline std::shared_ptr<aten::material> CreateMaterialWithParamter(
    std::string_view name,
    aten::context& ctxt,
    aten::MaterialType type,
    const aten::MaterialParameter& param)
{
    aten::MaterialParameter mtrl_param = param;
    mtrl_param.type = type;

    return ctxt.CreateMaterialWithMaterialParameter(
        name,
        mtrl_param,
        nullptr, nullptr, nullptr);
}

inline std::shared_ptr<aten::material> CreateMaterialWithParamterAndTextures(
    std::string_view name,
    aten::context& ctxt,
    aten::MaterialType type,
    const aten::MaterialParameter& param,
    aten::texture* albedoMap,
    aten::texture* normalMap,
    aten::texture* roughnessMap)
{
    aten::MaterialParameter mtrl_param = param;
    mtrl_param.type = type;

    return ctxt.CreateMaterialWithMaterialParameter(
        name,
        mtrl_param,
        albedoMap,
        normalMap,
        roughnessMap);
}
