#include "AssetManager.h"

#include <algorithm>

namespace aten {
    struct Asset {
        union {
            texture* tex;
            material* mtrl;
            object* obj;
        };

        AssetManager::AssetType type;

        Asset()
        {
            tex = nullptr;
        }
        Asset(texture* t) : tex(t), type(AssetManager::AssetType::Texture) {}
        Asset(material* m) : mtrl(m), type(AssetManager::AssetType::Material) {}
        Asset(object* o) : obj(o), type(AssetManager::AssetType::Object) {}

        bool operator==(const Asset& rhs) const
        {
            // NOTE
            // ポインタの比較なので、どれでもいい.
            return tex == rhs.tex;
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

    static bool registerAsset(
        const std::string& name,
        const Asset& asset,
        AssetManager::AssetType type)
    {
        auto& mapAsset = g_assets[type];

        auto it = mapAsset.find(name);
        if (it != mapAsset.end()) {
            AT_PRINTF("Registered already [%s] (%s)\n", name.c_str(), AssetTypeName[type]);
            return false;
        }

        mapAsset.insert(std::pair<std::string, Asset>(name, asset));

        return true;
    }

    static Asset& getAsset(
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

    static bool removeAsset(
        AssetManager::AssetType type,
        const Asset& asset)
    {
        auto found = std::find_if(
            g_assets[type].begin(),
            g_assets[type].end(),
            [&](std::pair<std::string, Asset> it)
        {
            return asset == it.second;
        });

        if (found != g_assets[type].end()) {
            g_assets[type].erase(found);
            return true;
        }

        return false;
    }

    bool AssetManager::registerMtrl(const std::string& name, material* mtrl)
    {
        mtrl->setName(name.c_str());

        return registerAsset(name, Asset(mtrl), AssetType::Material);
    }

    material* AssetManager::getMtrl(const std::string& name)
    {
        auto& asset = getAsset(name, AssetType::Material);
        return asset.mtrl;
    }

    bool AssetManager::removeMtrl(material* mtrl)
    {
        return removeAsset(AssetManager::AssetType::Material, Asset(mtrl));
    }

    bool AssetManager::registerTex(const std::string& name, texture* tex)
    {
        return registerAsset(name, Asset(tex), AssetType::Texture);
    }

    texture* AssetManager::getTex(const std::string& name)
    {
        auto& asset = getAsset(name, AssetType::Texture);
        return asset.tex;
    }

    bool AssetManager::removeTex(texture* tex)
    {
        return removeAsset(AssetManager::AssetType::Texture, Asset(tex));
    }

    bool AssetManager::registerObj(const std::string& name, object* obj)
    {
        return registerAsset(name, Asset(obj), AssetType::Object);
    }

    object* AssetManager::getObj(const std::string& name)
    {
        auto& asset = getAsset(name, AssetType::Object);
        return asset.obj;
    }

    bool AssetManager::removeObj(object* obj)
    {
        return removeAsset(AssetManager::AssetType::Object, Asset(obj));
    }

    void AssetManager::removeAllMtrls()
    {
        auto& assets = g_assets[AssetManager::AssetType::Material];

        for (auto it = assets.begin(); it != assets.end(); it++) {
            auto mtrl = it->second.mtrl;
            delete mtrl;
            assets.erase(it);
        }
    }

    void AssetManager::removeAllTextures()
    {
        auto& assets = g_assets[AssetManager::AssetType::Texture];

        for (auto it = assets.begin(); it != assets.end(); it++) {
            auto tex = it->second.tex;
            delete tex;
            assets.erase(it);
        }
    }

    void AssetManager::removeAllObjs()
    {
        auto& assets = g_assets[AssetManager::AssetType::Object];

        for (auto it = assets.begin(); it != assets.end(); it++) {
            auto obj = it->second.obj;
            delete obj;
            assets.erase(it);
        }
    }

    void AssetManager::suppressWarnings()
    {
        g_enableWarnings = false;
    }
}
