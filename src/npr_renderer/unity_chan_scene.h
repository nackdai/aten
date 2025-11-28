#pragma once

#include "aten.h"
#include "atenscene.h"

#include "../common/app_misc.h"
#include "../common/scenedefs.h"

class UnityChanScene {
public:
    static std::shared_ptr<aten::instance<aten::deformable>> makeScene(
        aten::context& ctxt, aten::scene* scene)
    {
        // NOTE:
        // The part of the normals in unitychan model seems to be broken.
        // That's the part of the hair.
        // It's not sure it happens from the original model or it happens while converting the model data.
        // If we re-compute it by computing cross the edges of the triangle, it's fixed.
        // But, it causes the non smooth normal.

        auto mdl = aten::TransformableFactory::createDeformable(ctxt);
        mdl->read("../../asset/converted_unitychan/unitychan_gpu.mdl");

        aten::ImageLoader::setBasePath("../../asset/unitychan/Texture");
        //aten::MaterialLoader::load("../../asset/converted_unitychan/unitychan_mtrl.xml", ctxt);
        aten::MaterialLoader::load("unitychan_toon_test_mtrl.xml", ctxt);

        for (auto& tex : ctxt.GetTextures()) {
            tex->SetFilterMode(aten::TextureFilterMode::Linear);
            tex->SetAddressMode(aten::TextureAddressMode::Wrap);
        }

        auto deformMdl = aten::TransformableFactory::createInstance<aten::deformable>(
            ctxt, mdl,
            aten::vec3(0), aten::vec3(0), aten::vec3(0.01F));
        scene->add(deformMdl);

        auto* mtrl_hair = ctxt.GetMaterialByName("hair");
        mtrl_hair->stencil_type = aten::StencilType::STENCIL;

        auto* mtrl_eyeline = ctxt.GetMaterialByName("eyeline");
        mtrl_eyeline->stencil_type = aten::StencilType::ALWAYS;
        mtrl_eyeline->feature_line.enable = false;

        auto* mtrl_cheek = ctxt.GetMaterialByName("mat_cheek");
        mtrl_cheek->feature_line.enable = false;

        auto* mtrl_eyebase = ctxt.GetMaterialByName("eyebase");
        mtrl_eyebase->feature_line.enable = false;

        auto* mtrl_eye_left = ctxt.GetMaterialByName("eye_L1");
        mtrl_eye_left->feature_line.enable = false;

        auto* mtrl_eye_right = ctxt.GetMaterialByName("eye_R1");
        mtrl_eye_right->feature_line.enable = false;

        aten::ImageLoader::load("FO_CLOTH1.tga", ctxt);

        auto* mtrl_face = ctxt.GetMaterialByName("face");
        AT_ASSERT(mtrl_face->type == aten::MaterialType::Toon);
        mtrl_face->toon.toon_type = aten::MaterialType::Diffuse;
        mtrl_face->toon.remap_texture = ctxt.GetTextureNum() - 1;
        mtrl_face->toon.target_light_idx = 0;
        mtrl_face->feature_line.metric_flag = aten::FeatureLineMetricFlag::Albedo | aten::FeatureLineMetricFlag::Normal | aten::FeatureLineMetricFlag::Depth;

        aten::ImageLoader::setBasePath("./");

        return deformMdl;
    }

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        float& fov)
    {
        pos = aten::vec3(0.f, 1.3f, 0.5f);
        at = aten::vec3(0.f, 1.3f, 0.f);
        fov = 45.0f;
    }
};
