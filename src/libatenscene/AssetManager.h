#pragma once

#include <map>
#include <memory>
#include <functional>
#include <optional>

#include "aten.h"

namespace aten {
    class AssetManager {
        friend class AssetRegister;

    public:
        enum class AssetType : int32_t {
            Texture,
            Material,
            Object,

            Invalid,

            Num,
        };

        AssetManager() = default;
        ~AssetManager() = default;

        AssetManager(const AssetManager&) = delete;
        AssetManager(AssetManager&&) = delete;
        AssetManager& operator=(const AssetManager&) = delete;
        AssetManager& operator=(AssetManager&&) = delete;

        bool registerMtrl(std::string_view name, const std::shared_ptr<material>& mtrl);
        std::shared_ptr<material> getMtrl(std::string_view name);
        std::shared_ptr<material> getMtrlByIdx(uint32_t idx);

        bool registerTex(std::string_view name, const std::shared_ptr<texture>& tex);
        std::shared_ptr<texture> getTex(const std::string& name);

        bool registerObj(std::string_view name, const std::shared_ptr<aten::PolygonObject>& obj);
        std::shared_ptr<aten::PolygonObject> getObj(std::string_view name);

        uint32_t getAssetNum(AssetType type);

    private:
        struct Asset {
            std::shared_ptr<texture> tex;
            std::shared_ptr<material> mtrl;
            std::shared_ptr<aten::PolygonObject> obj;

            AssetManager::AssetType type{ AssetManager::AssetType::Invalid };

            Asset(const std::shared_ptr<texture>& t)
                : tex(t), type(AssetManager::AssetType::Texture) {}
            Asset(const std::shared_ptr<material>& m)
                : mtrl(m), type(AssetManager::AssetType::Material) {}
            Asset(const std::shared_ptr<aten::PolygonObject>& o)
                : obj(o), type(AssetManager::AssetType::Object) {}

            Asset() {}
            ~Asset() {}

            Asset& operator=(const Asset&) = delete;
            Asset& operator=(Asset&&) = delete;

            Asset(const Asset& rhs)
            {
                type = rhs.type;

                switch (type) {
                case AssetManager::AssetType::Texture:
                    tex = rhs.tex;
                    break;
                case AssetManager::AssetType::Material:
                    mtrl = rhs.mtrl;
                    break;
                case AssetManager::AssetType::Object:
                    obj = rhs.obj;
                    break;
                default:
                    AT_ASSERT(false);
                    break;
                }
            }

            Asset(Asset&& rhs) noexcept
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
        };

        std::optional<Asset> getAsset(
            std::string_view name,
            AssetManager::AssetType type);

        std::array<std::map<std::string, Asset>, static_cast<size_t>(AssetManager::AssetType::Num)> assets_;
    };
}
