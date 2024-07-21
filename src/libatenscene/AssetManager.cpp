#include <algorithm>

#include "AssetManager.h"

namespace aten {
    static constexpr std::array<const char*, static_cast<size_t>(AssetManager::AssetType::Num)> AssetTypeName = {
        "Texture",
        "Material",
        "Object",
        "Invalid",
    };

    class AssetRegister {
    public:
        template <class T>
        static bool Register(
            AssetManager& asset_manager,
            std::string_view name,
            const std::shared_ptr<T>& asset,
            AssetManager::AssetType type)
        {
            const auto idx = static_cast<int32_t>(type);

            auto& mapAsset = asset_manager.assets_[idx];

            auto it = mapAsset.find(std::string(name));
            if (it != mapAsset.end()) {
                AT_PRINTF("Registered already [%s] (%s)\n", name.data(), AssetTypeName[idx]);
                return false;
            }

            mapAsset.insert(std::pair<std::string, AssetManager::Asset>(name, AssetManager::Asset(asset)));

            return true;
        }
    };

    std::optional<AssetManager::Asset> AssetManager::getAsset(
        std::string_view name,
        AssetManager::AssetType type)
    {
        auto& mapAsset = assets_[static_cast<int32_t>(type)];

        auto it = mapAsset.find(std::string(name));

        if (it == mapAsset.end()) {
            return std::nullopt;
        }

        return it->second;
    }

    bool AssetManager::registerObj(std::string_view name, const std::shared_ptr<aten::PolygonObject>& obj)
    {
        return AssetRegister::Register(*this, name, obj, AssetType::Object);
    }

    std::shared_ptr<aten::PolygonObject> AssetManager::getObj(std::string_view name)
    {
        auto asset = getAsset(name, AssetType::Object);
        if (asset) {
            return asset.value().obj;
        }
        return nullptr;
    }
}
