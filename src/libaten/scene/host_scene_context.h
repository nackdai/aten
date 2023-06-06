#pragma once

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <tuple>
#include <vector>

#include "geometry/vertex.h"
#include "texture/texture.h"
#include "visualizer/GeomDataBuffer.h"
#include "material/material.h"
#include "light/light_parameter.h"
#include "geometry/geomparam.h"
#include "math/mat4.h"

namespace AT_NAME {
    class triangle;
    class Light;
}

namespace aten
{
    class transformable;

    class context {
    public:
        context() = default;
        virtual ~context() = default;

    public:
        const aten::vec4 GetPosition(uint32_t idx) const noexcept
        {
            const auto& v = getVertex(idx);
            return aten::vec4(v.pos.x, v.pos.y, v.pos.z, v.uv.x);
        }

        const aten::vec4 GetPositionAsVec4(uint32_t idx) const noexcept
        {
            return GetPosition(idx);
        }

        const aten::vec3 GetPositionAsVec3(uint32_t idx) const noexcept
        {
            const auto& v = getVertex(idx);
            return aten::vec3(v.pos.x, v.pos.y, v.pos.z);
        }

        const aten::vec4 GetNormal(uint32_t idx) const noexcept
        {
            const auto& v = getVertex(idx);
            aten::vec4 res(v.nml, v.uv.y);
            return res;
        }

        const aten::vec4 GetNormalAsVec4(uint32_t idx) const noexcept
        {
            return GetNormal(idx);
        }

        const aten::vec3 GetNormalAsVec3(uint32_t idx) const noexcept
        {
            const auto& v = getVertex(idx);
            return v.nml;
        }

        const aten::ObjectParameter& GetObject(uint32_t idx) const noexcept;

        const aten::MaterialParameter& GetMaterial(uint32_t idx) const noexcept
        {
            return getMaterial(idx)->param();
        }

        const aten::TriangleParameter& GetTriangle(uint32_t idx) const noexcept;

        const aten::LightParameter& GetLight(uint32_t idx) const noexcept;

        const aten::mat4& GetMatrix(uint32_t idx) const noexcept
        {
            return *m_matrices[idx];
        }

        void addVertex(const aten::vertex& vtx)
        {
            m_vertices.push_back(vtx);
        }

        const aten::vertex& getVertex(int32_t idx) const
        {
            return m_vertices[idx];
        }

        uint32_t getVertexNum() const
        {
            return (uint32_t)m_vertices.size();
        }

        void copyVertices(std::vector<vertex>& dst) const
        {
            std::copy(
                m_vertices.begin(),
                m_vertices.end(),
                std::back_inserter(dst));
        }

        void build();

        const aten::GeomVertexBuffer& getVB() const
        {
            return m_vb;
        }

        void release()
        {
            m_vertices.clear();
            m_vb.clear();
        }

        std::shared_ptr<AT_NAME::material> createMaterial(
            aten::MaterialType type,
            aten::Values& value);

        std::shared_ptr<AT_NAME::material> createMaterialWithDefaultValue(
            aten::MaterialType type);

        std::shared_ptr<AT_NAME::material> createMaterialWithMaterialParameter(
            const aten::MaterialParameter& param,
            aten::texture* albedoMap,
            aten::texture* normalMap,
            aten::texture* roughnessMap);

        void addMaterial(std::shared_ptr<AT_NAME::material> mtrl)
        {
            AT_ASSERT(mtrl);
            m_materials.push_back(mtrl);
            mtrl->param().id = m_materials.size() - 1;
        }

        std::shared_ptr<AT_NAME::material> getMaterial(int32_t idx) const
        {
            AT_ASSERT(0 <= idx && idx < static_cast<int32_t>(m_materials.size()));
            return m_materials[idx];
        }

        void deleteAllMaterialsAndClearList();

        void copyMaterialParameters(std::vector<MaterialParameter>& dst) const;

        std::shared_ptr<const AT_NAME::material> findMaterialByName(const char* name) const;

        int32_t findMaterialIdxByName(const char* name) const;

        std::shared_ptr<AT_NAME::triangle> createTriangle(const aten::TriangleParameter& param);

        void addTriangle(const std::shared_ptr<AT_NAME::triangle>& tri);

        uint32_t getTriangleNum() const;

        std::shared_ptr<const AT_NAME::triangle> getTriangle(int32_t idx) const;

        void copyPrimitiveParameters(std::vector<aten::TriangleParameter>& dst) const;

        int32_t findTriIdxFromPointer(const void* p) const;

        void addTransformable(const std::shared_ptr<transformable>& t);

        std::shared_ptr<const aten::transformable> getTransformable(int32_t idx) const;

        void traverseTransformables(
            std::function<void(std::shared_ptr<aten::transformable>&, aten::ObjectType)> func) const;

        std::vector<aten::mat4> PickNonIndentityMatricesAndUpdateMtxidxOfInstance() const;

        int32_t findTransformableIdxFromPointer(const void* p) const;

        int32_t findPolygonalTransformableOrderFromPointer(const void* p) const;

        std::shared_ptr<texture> createTexture(
            uint32_t width, uint32_t height,
            uint32_t channels,
            std::string_view name);

        int32_t getTextureNum() const;

        std::shared_ptr<texture> getTexture(int32_t idx) const;

        void addTexture(const std::shared_ptr<texture>& tex);

        void initAllTexAsGLTexture();

        static void pinContext(const context* ctxt)
        {
            s_pinnedCtxt = ctxt;
        }

        static void removePinnedContext()
        {
            s_pinnedCtxt = nullptr;
        }

        static const context* getPinnedContext()
        {
            AT_ASSERT(s_pinnedCtxt);
            return s_pinnedCtxt;
        }

        std::tuple<uint32_t, std::shared_ptr<aten::mat4>> create_matrix()
        {
            size_t idx = m_matrices.size();
            m_matrices.reserve(idx + 1);
            m_matrices.push_back(std::make_shared<aten::mat4>());
            return std::make_tuple(static_cast<uint32_t>(idx), m_matrices[idx]);
        }

        void add_light(std::shared_ptr<AT_NAME::Light> light);
        std::shared_ptr<AT_NAME::Light> get_light(uint32_t idx) const;
        size_t get_light_num() const;

    private:
        static const context* s_pinnedCtxt;

        std::vector<aten::vertex> m_vertices;

        aten::GeomVertexBuffer m_vb;

        std::vector<std::shared_ptr<AT_NAME::material>> m_materials;
        std::vector<std::shared_ptr<AT_NAME::triangle>> m_triangles;
        mutable std::vector<std::shared_ptr<aten::transformable>> m_transformables;
        std::vector<std::shared_ptr<aten::texture>> m_textures;
        std::vector<std::shared_ptr<aten::mat4>> m_matrices;
        std::vector<std::shared_ptr<AT_NAME::Light>> m_lights;
    };
}
