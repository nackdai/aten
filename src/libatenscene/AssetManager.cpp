#include <algorithm>

#include "AssetManager.h"

namespace aten {
    struct Asset {
        std::shared_ptr<texture> tex;
        std::shared_ptr<material> mtrl;
        std::shared_ptr<aten::PolygonObject> obj;

        AssetManager::AssetType type;

        Asset() = default;
        ~Asset() = default;
        Asset(const std::shared_ptr<texture> t)
            : tex(t), type(AssetManager::AssetType::Texture) {}
        Asset(const std::shared_ptr<material> m)
            : mtrl(m), type(AssetManager::AssetType::Material) {}
        Asset(const std::shared_ptr<aten::PolygonObject> o)
            : obj(o), type(AssetManager::AssetType::Object) {}

        Asset(const Asset& rhs) = delete;
        Asset& operator=(const Asset& rhs) = delete;
        Asset& operator=(Asset&& rhs) = delete;

        Asset(Asset&& rhs)
        {
            type = rhs.type;

            switch (type) {
            case AssetManager::AssetType::Texture:
                tex = std::move(rhs.tex);
                break;
            case AssetManager::AssetType::Material:
                mtrl = std::move(rhs.mtrl);
                break;
            case AssetManager::AssetType::Object:
                obj = std::move(rhs.obj);
                break;
            default:
                AT_ASSERT(false);
                break;
            }
        }

        bool operator==(const Asset& rhs) const
        {
            bool ret = false;

            switch (type) {
            case AssetManager::AssetType::Texture:
                ret = tex == rhs.tex;
                break;
            case AssetManager::AssetType::Material:
                ret = mtrl == rhs.mtrl;
                break;
            case AssetManager::AssetType::Object:
                ret = obj == rhs.obj;
                break;
            default:
                AT_ASSERT(false);
                break;
            }

            return ret;
        }
    };

    using AssetStorage = std::map<std::string, Asset>;
    static AssetStorage g_assets[AssetManager::AssetType::Num];

    static bool g_enableWarnings = true;

    static const char* AssetTypeName[AssetManager::AssetType::Num] = {
        "Texture",
        "Material",
        "Object",
    };

    template <typename T>
    static bool registerAsset(
        const std::string& name,
        const std::shared_ptr<T> asset,
        AssetManager::AssetType type)
    {
        auto& mapAsset = g_assets[type];

        auto it = mapAsset.find(name);
        if (it != mapAsset.end()) {
            AT_PRINTF("Registered already [%s] (%s)\n", name.c_str(), AssetTypeName[type]);
            return false;
        }

        mapAsset.insert(std::pair<std::string, Asset>(name, Asset(asset)));

        return true;
    }

    static const Asset& getAsset(
        const std::string& name,
        AssetManager::AssetType type)
    {
        auto& mapAsset = g_assets[type];

        auto it = mapAsset.find(name);

        if (it == mapAsset.end()) {
            //AT_ASSERT(false);
            if (g_enableWarnings) {
                AT_PRINTF("Asset is not registered [%s] (%s)\n", name.c_str(), AssetTypeName[type]);
            }

            static Asset dummy;

            return dummy;
        }

        auto& asset = it->second;

        return asset;
    }

    bool AssetManager::registerMtrl(const std::string& name, const std::shared_ptr<material>& mtrl)
    {
        mtrl->setName(name.c_str());

        return registerAsset(name, mtrl, AssetType::Material);
    }

    std::shared_ptr<material> AssetManager::getMtrl(const std::string& name)
    {
        auto& asset = getAsset(name, AssetType::Material);
        return asset.mtrl;
    }

    std::shared_ptr<material> AssetManager::getMtrlByIdx(uint32_t idx)
    {
        const auto& assets = g_assets[AssetType::Material];
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

    bool AssetManager::registerTex(const std::string& name, const std::shared_ptr<texture>& tex)
    {
        return registerAsset(name, tex, AssetType::Texture);
    }

    std::shared_ptr<texture> AssetManager::getTex(const std::string& name)
    {
        auto& asset = getAsset(name, AssetType::Texture);
        return asset.tex;
    }

    bool AssetManager::registerObj(const std::string& name, const std::shared_ptr<aten::PolygonObject>& obj)
    {
        return registerAsset(name, obj, AssetType::Object);
    }

    std::shared_ptr<aten::PolygonObject> AssetManager::getObj(const std::string& name)
    {
        auto& asset = getAsset(name, AssetType::Object);
        return asset.obj;
    }

    uint32_t AssetManager::getAssetNum(AssetManager::AssetType type)
    {
        auto& assets = g_assets[type];
        return static_cast<uint32_t>(assets.size());
    }

    void AssetManager::suppressWarnings()
    {
        g_enableWarnings = false;
    }
}
