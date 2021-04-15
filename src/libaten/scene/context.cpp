#include <type_traits>

#include "scene/context.h"
#include "geometry/face.h"
#include "material/material_factory.h"
#include "geometry/transformable.h"

namespace aten
{
    const context* context::s_pinnedCtxt = nullptr;

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

    std::shared_ptr<AT_NAME::material> context::createMaterial(
        aten::MaterialType type,
        aten::Values& value)
    {
        auto mtrl = MaterialFactory::createMaterial(type, value);
        AT_ASSERT(mtrl);

        if (mtrl) {
            addMaterial(mtrl);
        }

        return mtrl;
    }

    std::shared_ptr<AT_NAME::material> context::createMaterialWithDefaultValue(aten::MaterialType type)
    {
        auto mtrl = MaterialFactory::createMaterialWithDefaultValue(type);
        AT_ASSERT(mtrl);

        if (mtrl) {
            addMaterial(mtrl);
        }

        return mtrl;
    }

    std::shared_ptr<AT_NAME::material> context::createMaterialWithMaterialParameter(
        const aten::MaterialParameter& param,
        aten::texture* albedoMap,
        aten::texture* normalMap,
        aten::texture* roughnessMap)
    {
        auto mtrl = MaterialFactory::createMaterialWithMaterialParameter(
            param,
            albedoMap,
            normalMap,
            roughnessMap);
        AT_ASSERT(mtrl);

        if (mtrl) {
            addMaterial(mtrl);
        }

        return mtrl;
    }

    void context::addMaterial(std::shared_ptr<AT_NAME::material>& mtrl)
    {
        AT_ASSERT(mtrl);
        m_materials.push_back(mtrl);
        mtrl->updateIndex(m_materials.size() - 1);
    }

    void context::addMaterial(AT_NAME::material* mtrl)
    {
        std::shared_ptr<std::remove_pointer<decltype(mtrl)>::type> m(mtrl);
        addMaterial(m);
    }

    void context::deleteAllMaterialsAndClearList()
    {
        m_materials.clear();
    }

    void context::copyMaterialParameters(std::vector<MaterialParameter>& dst) const
    {
        for (const auto& mtrl : m_materials) {
            dst.push_back(mtrl->param());
        }
    }

    const std::shared_ptr<AT_NAME::material>& context::findMaterialByName(const char* name) const
    {
        std::string strname(name);

        auto found = std::find_if(
            m_materials.begin(), m_materials.end(),
            [&](const std::shared_ptr<AT_NAME::material>& mtrl) {
                return mtrl->nameString() == strname;
            }
        );

        if (found != m_materials.end()) {
            const auto& mtrl = *found;
            return mtrl;
        }

        return nullptr;
    }

    int context::findMaterialIdxByName(const char* name) const
    {
        const auto& mtrl = findMaterialByName(name);
        if (mtrl) {
            return mtrl->id();
        }
        return -1;
    }

    AT_NAME::face* context::createTriangle(const aten::PrimitiveParamter& param)
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

    void context::addTransformable(aten::transformable* t)
    {
        t->addToDataList(m_transformables);
    }

    int context::getTransformableNum() const
    {
        return m_transformables.size();
    }

    const aten::transformable * context::getTransformable(int idx) const
    {
        AT_ASSERT(0 <= idx && idx < getTransformableNum());
        return m_transformables[idx];
    }

    void context::traverseTransformables(std::function<void(aten::transformable*, aten::GeometryType)> func) const
    {
        auto& shapes = m_transformables.getList();

        for (auto s : shapes) {
            auto t = s->getData();

            auto type = t->getType();

            func(t, type);
        }
    }

    void context::copyMatricesAndUpdateTransformableMatrixIdx(std::vector<aten::mat4>& dst) const
    {
        traverseTransformables([&](aten::transformable* t, aten::GeometryType type) {
            if (type == GeometryType::Instance) {
                aten::mat4 mtxL2W, mtxW2L;
                t->getMatrices(mtxL2W, mtxW2L);

                if (!mtxL2W.isIdentity()) {
                    auto& param = t->getParam();
                    param.mtxid = (int)(dst.size() / 2);

                    dst.push_back(mtxL2W);
                    dst.push_back(mtxW2L);
                }
            }
        });
    }

    int context::findTransformableIdxFromPointer(const void* p) const
    {
        auto& shapes = m_transformables.getList();

        auto found = std::find_if(
            shapes.begin(), shapes.end(),
            [&](const aten::DataList<aten::transformable>::ListItem* item) {
            const auto t = item->getData();
            return t == p;
        });

        int id = -1;

        if (found != shapes.end()) {
            const auto t = (*found)->getData();
            id = t->id();
        }

        return id;
    }

    int context::findPolygonalTransformableOrderFromPointer(const void* p) const
    {
        auto& shapes = m_transformables.getList();

        int order = -1;

        for (const auto item : shapes) {
            const auto t = item->getData();

            auto type = t->getType();
            if (type == aten::GeometryType::Polygon) {
                order++;
            }

            if (t == p) {
                break;
            }
        }

        AT_ASSERT(order >= 0);

        return order;
    }

    texture* context::createTexture(uint32_t width, uint32_t height, uint32_t channels, const char* name)
    {
        auto ret = texture::create(width, height, channels, name);
        AT_ASSERT(ret);

        addTexture(ret);

        return ret;
    }

    int context::getTextureNum() const
    {
        return m_textures.size();;
    }

    const texture* context::getTexture(int idx) const
    {
        AT_ASSERT(0 <= idx && idx < getTextureNum());
        return m_textures[idx];
    }

    texture* context::getTexture(int idx)
    {
        AT_ASSERT(0 <= idx && idx < getTextureNum());
        return m_textures[idx];
    }

    void context::addTexture(texture* tex)
    {
        AT_ASSERT(tex);

        tex->addToDataList(m_textures);
    }

    void context::initAllTexAsGLTexture()
    {
        auto num = getTextureNum();

        for (int i = 0; i < num; i++) {
            auto tex = m_textures[i];
            tex->initAsGLTexture();
        }
    }
}
