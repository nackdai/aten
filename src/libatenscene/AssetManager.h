#pragma once

#include <map>
#include <memory>
#include <functional>
#include "aten.h"

namespace aten {
    class AssetManager {
    private:
        AssetManager() = delete;
        ~AssetManager() = delete;

    public:
        enum AssetType {
            Texture,
            Material,
            Object,

            Num,
        };

        static bool registerMtrl(std::string_view name, const std::shared_ptr<material>& mtrl);
        static std::shared_ptr<material> getMtrl(std::string_view name);
        static std::shared_ptr<material> getMtrlByIdx(uint32_t idx);

        static bool registerTex(std::string_view name, const std::shared_ptr<texture>& tex);
        static std::shared_ptr<texture> getTex(const std::string& name);

        static bool registerObj(std::string_view name, const std::shared_ptr<aten::PolygonObject>& obj);
        static std::shared_ptr<aten::PolygonObject> getObj(std::string_view name);

        static uint32_t getAssetNum(AssetType type);

        static void suppressWarnings();
    };
}
