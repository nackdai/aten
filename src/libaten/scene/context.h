#pragma once

#include <algorithm>
#include <iterator>
#include <memory>
#include <vector>
#include <functional>
#include <tuple>

#include "geometry/vertex.h"
#include "visualizer/GeomDataBuffer.h"
#include "material/material.h"
#include "geometry/geomparam.h"
#include "geometry/transformable.h"
#include "texture/texture.h"
#include "math/mat4.h"

namespace AT_NAME {
    class triangle;
}

namespace aten
{
    class transformable;

    class context {
    public:
        context() = default;
        virtual ~context() = default;

    public:
        void addVertex(const aten::vertex& vtx)
        {
            m_vertices.push_back(vtx);
        }

        const aten::vertex& getVertex(int32_t idx) const
        {
            return m_vertices[idx];
        }

        aten::vertex& getVertex(int32_t idx)
        {
            return m_vertices[idx];
        }

        const std::vector<aten::vertex>& getVertices() const
        {
            return m_vertices;
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

        void addMaterial(const std::shared_ptr<AT_NAME::material>& mtrl);

        void addMaterial(AT_NAME::material* mtrl);

        int32_t getMaterialNum() const
        {
            return static_cast<int32_t>(m_materials.size());
        }

        std::shared_ptr<AT_NAME::material> getMaterial(int32_t idx) const
        {
            AT_ASSERT(0 <= idx && idx < getMaterialNum());
            return m_materials[idx];
        }

        void deleteAllMaterialsAndClearList();

        void copyMaterialParameters(std::vector<MaterialParameter>& dst) const;

        std::shared_ptr<const AT_NAME::material> findMaterialByName(const char* name) const;

        int32_t findMaterialIdxByName(const char* name) const;

        std::shared_ptr<AT_NAME::triangle> createTriangle(const aten::TriangleParameter& param);

        void addTriangle(const std::shared_ptr<AT_NAME::triangle>& tri);

        uint32_t getTriangleNum() const
        {
            return static_cast<uint32_t>(m_triangles.size());
        }

        std::shared_ptr<const AT_NAME::triangle> getTriangle(int32_t idx) const;

        void copyPrimitiveParameters(std::vector<aten::TriangleParameter>& dst) const;

        int32_t findTriIdxFromPointer(const void* p) const;

        auto addTransformable(const std::shared_ptr<transformable>& t)
        {
            AT_ASSERT(t);
            m_transformables.push_back(t);
            t->updateIndex(m_transformables.size() - 1);
        }

        std::shared_ptr<const aten::transformable> getTransformable(int32_t idx) const
        {
            AT_ASSERT(0 <= idx && idx < m_transformables.size());
            return m_transformables[idx];
        }

        void traverseTransformables(
            std::function<void(std::shared_ptr<aten::transformable>&, aten::ObjectType)> func) const;

        void pick_non_indentity_matrices(std::vector<aten::mat4>& dst) const;

        int32_t findTransformableIdxFromPointer(const void* p) const;

        int32_t findPolygonalTransformableOrderFromPointer(const void* p) const;

        std::shared_ptr<texture> createTexture(
            uint32_t width, uint32_t height, uint32_t channels, const char* name);

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

        const aten::ObjectParameter& get_object(uint32_t idx) const
        {
            return m_transformables[idx]->getParam();
        }
        const aten::ObjectParameter& get_real_object(const aten::ObjectParameter& obj) const
        {
            const auto& real_obj = obj.object_id >= 0 ? get_object(obj.object_id) : obj;
            return real_obj;
        }

        std::tuple<uint32_t, std::shared_ptr<aten::mat4>> create_matrix()
        {
            size_t idx = m_matrices.size();
            m_matrices.reserve(idx + 1);
            m_matrices.push_back(std::make_shared<aten::mat4>());
            return std::make_tuple(static_cast<uint32_t>(idx), m_matrices[idx]);
        }

        aten::mat4 get_matrix(uint32_t idx) const
        {
            if (idx >= m_matrices.size()) {
                return aten::mat4::Identity;
            }
            return *m_matrices[idx];
        }
        std::vector<aten::mat4> get_matrices()
        {
            std::vector<aten::mat4> mtxs;
            mtxs.reserve(m_matrices.size());
            for (const auto m : m_matrices) {
                mtxs.emplace_back(*m.get());
            }
            return mtxs;
        }

    private:
        static const context* s_pinnedCtxt;

        std::vector<aten::vertex> m_vertices;

        aten::GeomVertexBuffer m_vb;

        std::vector<std::shared_ptr<AT_NAME::material>> m_materials;
        std::vector<std::shared_ptr<AT_NAME::triangle>> m_triangles;
        mutable std::vector<std::shared_ptr<aten::transformable>> m_transformables;
        std::vector<std::shared_ptr<aten::texture>> m_textures;
        std::vector<std::shared_ptr<aten::mat4>> m_matrices;
    };
}
