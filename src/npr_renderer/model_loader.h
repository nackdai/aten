#pragma once

#include "atenscene.h"

class ModelLoader {
private:
    ModelLoader() = default;
    ~ModelLoader() = default;

public:
    static void Load(
        std::vector<std::shared_ptr<aten::PolygonObject>>& objs,
        aten::context& ctxt,
        std::string_view objpath)
    {
        Load(objs, ctxt, objpath, "");

        const auto mtrl_num = ctxt.GetMaterialNum();
        for (size_t i = 0; i < mtrl_num; i++) {
            auto mtrl = ctxt.GetMaterialInstance(i);
            std::string tex_name = "remap_" + mtrl->nameString();
            auto tex = ctxt.CreateTexture(256, 1, 3, tex_name, aten::vec4(1.0F));

            mtrl->param().toon.remap_texture = tex->id();
            mtrl->param().toon.target_light_idx = 0;
        }
    }

private:
    static void Load(
        std::vector<std::shared_ptr<aten::PolygonObject>>& objs,
        aten::context& ctxt,
        std::string_view objpath,
        std::string_view mtrlpath)
    {
        std::string pathname;
        std::string extname;
        std::string filename;

        aten::getStringsFromPath(
            objpath,
            pathname,
            extname,
            filename);

        if (pathname[pathname.size() - 1] != '/') {
            pathname += '/';
        }

        if (!mtrlpath.empty()) {
            aten::MaterialLoader::load(mtrlpath, ctxt);
        }

        aten::AssimpImporter::load(
            objpath,
            objs,
            ctxt,
            [&](std::string_view name,
                aten::context& ctxt,
                const aten::MaterialParameter& mtrl_param,
                const std::string& albedo,
                const std::string& nml)
            {
                auto mtrl = ctxt.FindMaterialByName(name);
                if (!mtrl) {
                    auto albedo_map = albedo.empty()
                        ? nullptr
                        : aten::ImageLoader::load(pathname + albedo, ctxt);
                    auto nml_map = nml.empty()
                        ? nullptr
                        : aten::ImageLoader::load(pathname + nml, ctxt);

                    auto toon_mtrl_param = mtrl_param;
                    toon_mtrl_param.type = aten::MaterialType::Toon;

                    mtrl = ctxt.CreateMaterialWithMaterialParameter(
                        name,
                        toon_mtrl_param,
                        albedo_map.get(),
                        nml_map.get(),
                        nullptr);
                }

                return mtrl;
            });
    }
};

class DummyScene {
public:
    static std::shared_ptr<aten::instance<aten::deformable>> makeScene(
        aten::context& ctxt, aten::scene* scene)
    {
        return nullptr;
    }

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        float& fov)
    {
        pos = aten::vec3(-0.026f, 17.4402f, 2.7454f);
        at = aten::vec3(-0.026f, 17.4402f, 1.7454f);
        fov = 45.0f;
    }
};
