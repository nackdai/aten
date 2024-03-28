#include <algorithm>

#include "AssetManager.h"

namespace aten {
    static constexpr std::array<char*, static_cast<size_t>(AssetManager::AssetType::Num)> AssetTypeName = {
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

    bool AssetManager::registerMtrl(std::string_view name, const std::shared_ptr<material>& mtrl)
    {
        mtrl->setName(name);

        return AssetRegister::Register(*this, name, mtrl, AssetType::Material);
    }

    std::shared_ptr<material> AssetManager::getMtrl(std::string_view name)
    {
        auto asset = getAsset(name, AssetType::Material);
        if (asset) {
            return asset.value().mtrl;
        }
        return nullptr;
    }

    std::shared_ptr<material> AssetManager::getMtrlByIdx(uint32_t idx)
    {
        const auto& assets = assets_[static_cast<int32_t>(AssetType::Material)];
        if (idx < assets.size()) {
            uint32_t pos = 0;
            for (auto it = assets.begin(); it != assets.end(); it++, pos++) {
                if (pos == idx) {
                    return it->second.mtrl;
                }
            }
        }

        AT_ASSERT(false);
        return nullptr;
    }

    bool AssetManager::registerTex(std::string_view name, const std::shared_ptr<texture>& tex)
    {
        return AssetRegister::Register(*this, name, tex, AssetType::Texture);
    }

    std::shared_ptr<texture> AssetManager::getTex(const std::string& name)
    {
        auto& asset = getAsset(name, AssetType::Texture);
        if (asset) {
            return asset.value().tex;
        }
        return nullptr;
    }

    bool AssetManager::registerObj(std::string_view name, const std::shared_ptr<aten::PolygonObject>& obj)
    {
        return AssetRegister::Register(*this, name, obj, AssetType::Object);
    }

    std::shared_ptr<aten::PolygonObject> AssetManager::getObj(std::string_view name)
    {
        auto& asset = getAsset(name, AssetType::Object);
        if (asset) {
            return asset.value().obj;
        }
        return nullptr;
    }

    uint32_t AssetManager::getAssetNum(AssetManager::AssetType type)
    {
        auto& assets = assets_[static_cast<int32_t>(type)];
        return static_cast<uint32_t>(assets.size());
    }
}
