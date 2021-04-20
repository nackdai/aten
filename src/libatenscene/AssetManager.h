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

        static bool registerMtrl(const std::string& name, const std::shared_ptr<material> mtrl);
        static const std::shared_ptr<material> getMtrl(const std::string& name);
        static const std::shared_ptr<material> getMtrlByIdx(uint32_t idx);

        static bool registerTex(const std::string& name, std::shared_ptr<texture> tex);
        static const std::shared_ptr<texture> getTex(const std::string& name);

        static bool registerObj(const std::string& name, const std::shared_ptr<object> obj);
        static const std::shared_ptr<object> getObj(const std::string& name);

        static uint32_t getAssetNum(AssetType type);

        static void suppressWarnings();
    };
}
