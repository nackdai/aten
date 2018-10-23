#include "scene/context.h"
#include "geometry/face.h"

namespace aten
{
    void context::build()
    {
        if (!m_vertices.empty()
            && !m_vb.isInitialized())
        {
            m_vb.init(
                sizeof(vertex),
                m_vertices.size(),
                0,
                &m_vertices[0]);
        }
    }

    void context::addMaterial(aten::material* mtrl)
    {
        AT_ASSERT(mtrl);
        mtrl->addToDataList(m_materials);
    }

    void context::deleteAllMaterialsAndClearList()
    {
        m_materials.deleteAllDataAndClear();
    }

    void context::copyMaterialParameters(std::vector<MaterialParameter>& dst) const
    {
        auto& materials = m_materials.getList();

        for (const auto item : materials) {
            auto mtrl = item->getData();
            dst.push_back(mtrl->param());
        }
    }

    const material* context::findMaterialByName(const char* name) const
    {
        std::string strname(name);

        auto& materials = m_materials.getList();

        auto found = std::find_if(
            materials.begin(), materials.end(),
            [&](const aten::DataList<aten::material>::ListItem* item) {
            auto mtrl = item->getData();
            return mtrl->nameString() == strname;
        });

        if (found != materials.end()) {
            auto* item = *found;
            auto mtrl = item->getData();
            return mtrl;
        }

        return nullptr;
    }

    int context::findMaterialIdxByName(const char* name) const
    {
        auto mtrl = findMaterialByName(name);
        if (mtrl) {
            return mtrl->id();
        }
        return -1;
    }

    AT_NAME::face * context::createTriangle(const PrimitiveParamter & param)
    {
        auto f = AT_NAME::face::create(*this, param);
        AT_ASSERT(f);
        
        if (f) {
            addTriangle(f);
        }

        return f;
    }

    void context::addTriangle(AT_NAME::face* tri)
    {
        tri->addToDataList(m_triangles);
    }

    int context::getTriangleNum() const
    {
        return m_triangles.size();
    }

    const AT_NAME::face* context::getTriangle(int idx) const
    {
        AT_ASSERT(0 <= idx && idx < getTriangleNum());
        return m_triangles[idx];
    }

    void context::copyPrimitiveParameters(std::vector<aten::PrimitiveParamter>& dst) const
    {
        auto& triangles = m_triangles.getList();

        for (const auto item : triangles) {
            const auto tri = item->getData();
            dst.push_back(tri->getParam());
        }
    }

    int context::findTriIdxFromPointer(const void* p) const
    {
        auto& triangles = m_triangles.getList();

        auto found = std::find_if(
            triangles.begin(), triangles.end(),
            [&](const aten::DataList<aten::face>::ListItem* item) {
            const auto tri = item->getData();
            return tri == p;
        });

        int id = -1;

        if (found != triangles.end()) {
            const auto tri = (*found)->getData();
            id = tri->getId();
        }

        return id;
    }
}
