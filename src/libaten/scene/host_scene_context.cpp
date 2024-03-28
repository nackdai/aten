#include <type_traits>

#include "scene/host_scene_context.h"
#include "geometry/triangle.h"
#include "geometry/transformable.h"
#include "light/light.h"

namespace aten
{
    const context* context::s_pinnedCtxt = nullptr;

    const aten::ObjectParameter& context::GetObject(uint32_t idx) const noexcept
    {
        return GetTransformable(idx)->GetParam();
    }

    const aten::TriangleParameter& context::GetTriangle(uint32_t idx) const noexcept
    {
        return triangles_[idx]->GetParam();
    }

    const aten::LightParameter& context::GetLight(uint32_t idx) const noexcept
    {
        return lights_[idx]->param();
    }

    void context::build()
    {
        if (!vertices_.empty()
            && !vertex_buffer_.isInitialized())
        {
            vertex_buffer_.init(
                sizeof(vertex),
                static_cast<uint32_t>(vertices_.size()),
                0,
                &vertices_[0]);
        }
    }

    std::shared_ptr<AT_NAME::material> context::CreateMaterial(
        aten::MaterialType type,
        aten::Values& value)
    {
        auto mtrl = material::CreateMaterial(type, value);
        AT_ASSERT(mtrl);

        if (mtrl) {
            AddMaterial(mtrl);
        }

        return mtrl;
    }

    std::shared_ptr<AT_NAME::material> context::CreateMaterialWithDefaultValue(aten::MaterialType type)
    {
        auto mtrl = material::CreateMaterialWithDefaultValue(type);
        AT_ASSERT(mtrl);

        if (mtrl) {
            AddMaterial(mtrl);
        }

        return mtrl;
    }

    std::shared_ptr<AT_NAME::material> context::CreateMaterialWithMaterialParameter(
        const aten::MaterialParameter& param,
        aten::texture* albedoMap,
        aten::texture* normalMap,
        aten::texture* roughnessMap)
    {
        auto mtrl = material::CreateMaterialWithMaterialParameter(
            param,
            albedoMap,
            normalMap,
            roughnessMap);
        AT_ASSERT(mtrl);

        if (mtrl) {
            AddMaterial(mtrl);
        }

        return mtrl;
    }

    void context::DeleteAllMaterialsAndClearList()
    {
        materials_.clear();
    }

    void context::CopyMaterialParameters(std::vector<MaterialParameter>& dst) const
    {
        for (const auto& mtrl : materials_) {
            dst.push_back(mtrl->param());
        }
    }

    std::shared_ptr<const AT_NAME::material> context::FindMaterialByName(std::string_view name) const
    {
        std::string strname(name);

        auto found = std::find_if(
            materials_.begin(), materials_.end(),
            [&](const std::shared_ptr<AT_NAME::material> mtrl) {
                return mtrl->nameString() == strname;
            }
        );

        if (found != materials_.end()) {
            const auto& mtrl = *found;
            return mtrl;
        }

        return nullptr;
    }

    int32_t context::FindMaterialIdxByName(std::string_view name) const
    {
        auto mtrl = FindMaterialByName(name);
        if (mtrl) {
            return mtrl->id();
        }
        return -1;
    }

    std::shared_ptr<AT_NAME::triangle> context::CreateTriangle(const aten::TriangleParameter& param)
    {
        auto f = AT_NAME::triangle::create(*this, param);
        AT_ASSERT(f);

        if (f) {
            AddTriangle(f);
        }

        return f;
    }

    void context::AddTriangle(const std::shared_ptr<AT_NAME::triangle>& tri)
    {
        AT_ASSERT(tri);
        triangles_.push_back(tri);
        tri->updateIndex(triangles_.size() - 1);
    }

    uint32_t context::GetTriangleNum() const
    {
        return static_cast<uint32_t>(triangles_.size());
    }

    std::shared_ptr<const AT_NAME::triangle> context::GetTriangleInstance(int32_t idx) const
    {
        AT_ASSERT(0 <= idx && idx < triangles_.size());
        return triangles_[idx];
    }

    void context::CopyTriangleParameters(std::vector<aten::TriangleParameter>& dst) const
    {
        for (const auto& tri : triangles_) {
            dst.push_back(tri->GetParam());
        }
    }

    int32_t context::FindTriangleIdxFromPointer(const void* p) const
    {
        auto found = std::find_if(
            triangles_.begin(), triangles_.end(),
            [&](const std::shared_ptr<AT_NAME::triangle> triangle) {
            return triangle.get() == p;
        });

        int32_t id = -1;

        if (found != triangles_.end()) {
            const auto& tri = *found;
            id = tri->GetId();
        }

        return id;
    }

    void context::AddTransformable(const std::shared_ptr<transformable>& t)
    {
        AT_ASSERT(t);
        transformables_.push_back(t);
        t->updateIndex(transformables_.size() - 1);
    }

    std::shared_ptr<const aten::transformable> context::GetTransformable(int32_t idx) const
    {
        AT_ASSERT(0 <= idx && idx < transformables_.size());
        return transformables_[idx];
    }

    void context::TraverseTransformables(
        std::function<void(std::shared_ptr<aten::transformable>&, aten::ObjectType)> func) const
    {
        for (auto& t : transformables_) {
            auto type = t->getType();
            func(t, type);
        }
    }

    std::vector<aten::mat4> context::PickNonIdentityMatricesAndUpdateMatrixIdxInTransformable() const
    {
        std::vector<aten::mat4> dst;
        TraverseTransformables([&dst](std::shared_ptr<aten::transformable>& t, aten::ObjectType type) {
            if (type == ObjectType::Instance) {
                aten::mat4 mtx_L2W, mtx_W2L;
                t->getMatrices(mtx_L2W, mtx_W2L);

                auto& param = t->GetParam();

                if (mtx_L2W.isIdentity()) {
                    param.mtx_id = -1;
                }
                else {
                    param.mtx_id = (int32_t)(dst.size() / 2);

                    dst.push_back(mtx_L2W);
                    dst.push_back(mtx_W2L);
                }
            }
        });
        return dst;
    }

    int32_t context::FindTransformableIdxFromPointer(const void* p) const
    {
        auto found = std::find_if(
            transformables_.begin(), transformables_.end(),
            [&](const std::shared_ptr<aten::transformable>& t) {
            return t.get() == p;
        });

        int32_t id = -1;

        if (found != transformables_.end()) {
            const auto& t = *found;
            id = t->id();
        }

        return id;
    }

    int32_t context::FindPolygonalTransformableOrderFromPointer(const void* p) const
    {
        int32_t order = -1;

        for (const auto& t : transformables_) {
            auto type = t->getType();
            if (type == aten::ObjectType::Polygons) {
                order++;
            }

            if (t.get() == p) {
                break;
            }
        }

        AT_ASSERT(order >= 0);

        return order;
    }

    std::shared_ptr<texture> context::CreateTexture(
        int32_t width, int32_t height,
        uint32_t channels,
        std::string_view name)
    {
        auto ret = texture::create(width, height, channels, name);
        AT_ASSERT(ret);

        AddTexture(ret);

        return ret;
    }

    std::shared_ptr<texture> context::GtTexture(int32_t idx) const
    {
        AT_ASSERT(0 <= idx && idx < GetTextureNum());
        return textures_[idx];
    }

    void context::AddTexture(const std::shared_ptr<texture>& tex)
    {
        AT_ASSERT(tex);
        textures_.push_back(tex);
        tex->updateIndex(textures_.size() - 1);
    }

    void context::InitAllTextureAsGLTexture()
    {
        auto num = GetTextureNum();

        for (int32_t i = 0; i < num; i++) {
            auto tex = textures_[i];
            tex->initAsGLTexture();
        }
    }

    void context::AddLight(const std::shared_ptr<Light>& light)
    {
        lights_.emplace_back(light);
    }

    std::shared_ptr<Light> context::GetLightInstance(uint32_t idx) const
    {
        return lights_[idx];
    }

    size_t context::GetLightNum() const
    {
        return lights_.size();
    }
}
