#pragma once

#include <vector>
#include <algorithm>
#include <iterator>

#include "geometry/vertex.h"
#include "visualizer/GeomDataBuffer.h"
#include "material/material.h"
#include "misc/datalist.h"
#include "geometry/geomparam.h"

namespace AT_NAME {
    class face;
}

namespace aten
{
    class transformable;

    class context {
    public:
        context() {}
        virtual ~context() {}

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

        AT_NAME::material* createMaterial(
            aten::MaterialType type,
            aten::Values& value);

        AT_NAME::material* createMaterialWithDefaultValue(aten::MaterialType type);

        AT_NAME::material* createMaterialWithMaterialParameter(
            aten::MaterialType type,
            const aten::MaterialParameter& param,
            aten::texture* albedoMap,
            aten::texture* normalMap,
            aten::texture* roughnessMap);

        void addMaterial(AT_NAME::material* mtrl);

        int getMaterialNum() const
        {
            return static_cast<int>(m_materials.size());
        }

        AT_NAME::material* getMaterial(int idx)
        {
            AT_ASSERT(0 <= idx && idx < getMaterialNum());
            return m_materials[idx];
        }

        const AT_NAME::material* getMaterial(int idx) const
        {
            AT_ASSERT(0 <= idx && idx < getMaterialNum());
            return m_materials[idx];
        }

        void deleteAllMaterialsAndClearList();

        void copyMaterialParameters(std::vector<MaterialParameter>& dst) const;

        const AT_NAME::material* findMaterialByName(const char* name) const;

        int findMaterialIdxByName(const char* name) const;

        AT_NAME::face* createTriangle(const aten::PrimitiveParamter& param);

        void addTriangle(AT_NAME::face* tri);

        int getTriangleNum() const;

        const AT_NAME::face* getTriangle(int idx) const;

        void copyPrimitiveParameters(std::vector<aten::PrimitiveParamter>& dst) const;

        int findTriIdxFromPointer(const void* p) const;

        void addTransformable(aten::transformable* t);

        int getTransformableNum() const;

        const aten::transformable* getTransformable(int idx) const;

        void traverseTransformables(std::function<void(aten::transformable*, aten::GeometryType)> func) const;

        void copyMatricesAndUpdateTransformableMatrixIdx(std::vector<aten::mat4>& dst) const;

        int findTransformableIdxFromPointer(const void* p) const;

        int findPolygonalTransformableOrderFromPointer(const void* p) const;

    private:
        std::vector<aten::vertex> m_vertices;

        aten::GeomVertexBuffer m_vb;

        DataList<AT_NAME::material> m_materials;
        DataList<AT_NAME::face> m_triangles;
        DataList<aten::transformable> m_transformables;
    };
}
