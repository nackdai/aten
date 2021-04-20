#pragma once

#include <algorithm>
#include <iterator>
#include <memory>
#include <vector>

#include "geometry/vertex.h"
#include "visualizer/GeomDataBuffer.h"
#include "material/material.h"
#include "misc/datalist.h"
#include "geometry/geomparam.h"
#include "texture/texture.h"

namespace AT_NAME {
    class face;
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

        const aten::vertex& getVertex(int idx) const
        {
            return m_vertices[idx];
        }

        aten::vertex& getVertex(int idx)
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

        void addMaterial(std::shared_ptr<AT_NAME::material> mtrl);

        void addMaterial(AT_NAME::material* mtrl);

        int getMaterialNum() const
        {
            return static_cast<int>(m_materials.size());
        }

        std::shared_ptr<AT_NAME::material> getMaterial(int idx)
        {
            AT_ASSERT(0 <= idx && idx < getMaterialNum());
            return m_materials[idx];
        }

        const std::shared_ptr<AT_NAME::material> getMaterial(int idx) const
        {
            AT_ASSERT(0 <= idx && idx < getMaterialNum());
            return m_materials[idx];
        }

        void deleteAllMaterialsAndClearList();

        void copyMaterialParameters(std::vector<MaterialParameter>& dst) const;

        const std::shared_ptr<AT_NAME::material> findMaterialByName(const char* name) const;

        int findMaterialIdxByName(const char* name) const;

        std::shared_ptr<AT_NAME::face> createTriangle(const aten::PrimitiveParamter& param);

        void addTriangle(std::shared_ptr<AT_NAME::face> tri);

        int getTriangleNum() const;

        const std::shared_ptr<AT_NAME::face> getTriangle(int idx) const;

        void copyPrimitiveParameters(std::vector<aten::PrimitiveParamter>& dst) const;

        int findTriIdxFromPointer(const void* p) const;

        void addTransformable(std::shared_ptr<aten::transformable> t);

        int getTransformableNum() const;

        const std::shared_ptr<aten::transformable> getTransformable(int idx) const;

        void traverseTransformables(
            std::function<void(const std::shared_ptr<aten::transformable>, aten::GeometryType)> func) const;

        void copyMatricesAndUpdateTransformableMatrixIdx(std::vector<aten::mat4>& dst) const;

        int findTransformableIdxFromPointer(const void* p) const;

        int findPolygonalTransformableOrderFromPointer(const void* p) const;

        std::shared_ptr<texture> createTexture(
            uint32_t width, uint32_t height, uint32_t channels, const char* name);

        int getTextureNum() const;

        const std::shared_ptr<texture> getTexture(int idx) const;
        std::shared_ptr<texture> getTexture(int idx);

        void addTexture(std::shared_ptr<texture> tex);

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

    private:
        static const context* s_pinnedCtxt;

        std::vector<aten::vertex> m_vertices;

        aten::GeomVertexBuffer m_vb;

        std::vector<std::shared_ptr<AT_NAME::material>> m_materials;
        std::vector<std::shared_ptr<AT_NAME::face>> m_triangles;
        std::vector<std::shared_ptr<aten::transformable>> m_transformables;
        std::vector<std::shared_ptr<aten::texture>> m_textures;
    };
}
