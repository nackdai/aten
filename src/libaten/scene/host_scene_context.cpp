#include <type_traits>

#include <nanovdb/NanoVDB.h>

#include "scene/host_scene_context.h"
#include "geometry/triangle.h"
#include "geometry/transformable.h"
#include "light/light.h"
#include "scene/scene.h"
#include "volume/grid.h"

namespace aten
{
    const context* context::s_pinnedCtxt = nullptr;

    const aten::ObjectParameter& context::GetObject(uint32_t idx) const noexcept
    {
        return GetTransformable(idx)->GetParam();
    }

    aten::tuple<std::vector<aten::ObjectParameter>, std::vector<aten::mat4>> context::GetObjectParametersAndMatrices() const
    {
        // NOTE
        // Apply the index to the tranform matrix of the objects in the following.
        // So, the following have to be called before aggrigating the object paraemters.
        auto mtxs = PickNonIdentityMatricesAndUpdateMatrixIdxInTransformable();

        std::vector<aten::ObjectParameter> objs;
        objs.reserve(transformables_.size());

        for (const auto& transformable : transformables_) {
            objs.emplace_back(transformable->GetParam());
        }

        return aten::make_tuple(objs, mtxs);
    }

    std::vector<aten::MaterialParameter> context::GetMetarialParemeters() const
    {
        std::vector<aten::MaterialParameter> mtrls;
        mtrls.reserve(materials_.size());

        for (const auto& material : materials_) {
            mtrls.emplace_back(material->param());
        }

        return mtrls;
    }

    const aten::TriangleParameter& context::GetTriangle(uint32_t idx) const noexcept
    {
        return triangles_[idx]->GetParam();
    }

    const aten::LightParameter& context::GetLight(uint32_t idx) const noexcept
    {
        return lights_[idx]->param();
    }

    std::vector<aten::LightParameter> context::GetLightParameters() const
    {
        std::vector<aten::LightParameter> lights;
        lights.reserve(lights_.size());

        for (const auto& light : lights_) {
            lights.emplace_back(light->param());
        }

        return lights;
    }

    aten::tuple<std::vector<aten::vec4>, std::vector<aten::vec4>> context::GetExtractedPosAndNmlInVertices() const
    {
        std::vector<aten::vec4> positions;
        std::vector<aten::vec4> normals;

        positions.reserve(vertices_.size());
        normals.reserve(vertices_.size());

        for (const auto& v : vertices_) {
            positions.emplace_back(aten::vec4(v.pos.x, v.pos.y, v.pos.z, v.uv.x));
            normals.emplace_back(aten::vec4(v.nml.x, v.nml.y, v.nml.z, v.uv.y));
        }

        return aten::make_tuple(positions, normals);
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
        std::string_view name,
        aten::MaterialType type,
        aten::Values& value)
    {
        auto mtrl = FindMaterialByName(name);

        if (!mtrl) {
            mtrl = material::CreateMaterial(type, value);
            AT_ASSERT(mtrl);

            if (mtrl) {
                AddMaterial(mtrl);
                mtrl->setName(name);
            }
        }

        return mtrl;
    }

    std::shared_ptr<AT_NAME::material> context::CreateMaterialWithMaterialParameter(
        std::string_view name,
        const aten::MaterialParameter& param,
        aten::texture* albedoMap,
        aten::texture* normalMap,
        aten::texture* roughnessMap)
    {
        auto mtrl = FindMaterialByName(name);

        if (!mtrl) {
            mtrl = material::CreateMaterialWithMaterialParameter(
                param,
                albedoMap,
                normalMap,
                roughnessMap);
            AT_ASSERT(mtrl);

            if (mtrl) {
                AddMaterial(mtrl);
                mtrl->setName(name);
            }
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

    std::shared_ptr<AT_NAME::triangle> context::GetTriangleInstance(int32_t idx) const
    {
        AT_ASSERT(0 <= idx && idx < triangles_.size());
        return triangles_[idx];
    }

    std::vector<aten::TriangleParameter> context::GetPrimitiveParameters() const
    {
        std::vector<aten::TriangleParameter> prims;
        prims.reserve(triangles_.size());

        for (const auto& triangle : triangles_) {
            prims.emplace_back(triangle->GetParam());
        }

        return prims;
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

    std::shared_ptr<aten::transformable> context::GetTransformable(int32_t idx) const
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
        auto tex = GetTextureByName(name);
        if (!tex) {
            tex = texture::create(width, height, channels, name);
            AT_ASSERT(tex);

            AddTexture(tex);
        }

        return tex;
    }

    std::shared_ptr<texture> context::GetTexture(int32_t idx) const
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

        // Assigne light id to object of light.
        if (light->param().arealight_objid >= 0) {
            const auto light_id = lights_.size() - 1;
            auto obj = transformables_.at(light->param().arealight_objid);
            obj->GetParam().light_id = light_id;

            if (obj->GetParam().type == aten::ObjectType::Instance) {
                obj = transformables_.at(obj->GetParam().object_id);
                obj->GetParam().light_id = light_id;
            }
        }
    }

    std::shared_ptr<Light> context::GetLightInstance(uint32_t idx) const
    {
        return lights_[idx];
    }

    size_t context::GetLightNum() const
    {
        return lights_.size();
    }

    void context::CleanAll()
    {
        vertices_.clear();
        vertex_buffer_.clear();
        materials_.clear();
        triangles_.clear();
        transformables_.clear();
        textures_.clear();
        matrices_.clear();
        lights_.clear();
        if (grid_holder_) {
            grid_holder_->Clear();
        }
    }

    void context::UpdateSceneBoundingBox(const AT_NAME::scene& scene)
    {
        scene_bounding_box_ = scene.GetBoundingBox();
    }

    const AT_NAME::Grid* context::GetGrid() const noexcept
    {
        return grid_holder_.get();
    }

    void context::RegisterGridHolder(const std::shared_ptr<AT_NAME::Grid>& grid_holder)
    {
        grid_holder_ = grid_holder;
    }
}
