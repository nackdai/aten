#include <type_traits>

#include "scene/context.h"
#include "geometry/face.h"
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
        auto mtrl = material::createMaterial(type, value);
        AT_ASSERT(mtrl);

        if (mtrl) {
            addMaterial(mtrl);
        }

        return mtrl;
    }

    std::shared_ptr<AT_NAME::material> context::createMaterialWithDefaultValue(aten::MaterialType type)
    {
        auto mtrl = material::createMaterialWithDefaultValue(type);
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
        auto mtrl = material::createMaterialWithMaterialParameter(
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

    void context::addMaterial(const std::shared_ptr<AT_NAME::material>& mtrl)
    {
        AT_ASSERT(mtrl);
        m_materials.push_back(mtrl);
        mtrl->param().id = m_materials.size() - 1;
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

    std::shared_ptr<const AT_NAME::material> context::findMaterialByName(const char* name) const
    {
        std::string strname(name);

        auto found = std::find_if(
            m_materials.begin(), m_materials.end(),
            [&](const std::shared_ptr<AT_NAME::material> mtrl) {
                return mtrl->nameString() == strname;
            }
        );

        if (found != m_materials.end()) {
            const auto& mtrl = *found;
            return mtrl;
        }

        return nullptr;
    }

    int32_t context::findMaterialIdxByName(const char* name) const
    {
        auto mtrl = findMaterialByName(name);
        if (mtrl) {
            return mtrl->id();
        }
        return -1;
    }

    std::shared_ptr<AT_NAME::face> context::createTriangle(const aten::TriangleParameter& param)
    {
        auto f = AT_NAME::face::create(*this, param);
        AT_ASSERT(f);

        if (f) {
            addTriangle(f);
        }

        return f;
    }

    void context::addTriangle(const std::shared_ptr<AT_NAME::face>& tri)
    {
        AT_ASSERT(tri);
        m_triangles.push_back(tri);
        tri->updateIndex(m_triangles.size() - 1);
    }

    std::shared_ptr<const AT_NAME::face> context::getTriangle(int32_t idx) const
    {
        AT_ASSERT(0 <= idx && idx < m_triangles.size());
        return m_triangles[idx];
    }

    void context::copyPrimitiveParameters(std::vector<aten::TriangleParameter>& dst) const
    {
        for (const auto& tri : m_triangles) {
            dst.push_back(tri->getParam());
        }
    }

    int32_t context::findTriIdxFromPointer(const void* p) const
    {
        auto found = std::find_if(
            m_triangles.begin(), m_triangles.end(),
            [&](const std::shared_ptr<AT_NAME::face> face) {
            return face.get() == p;
        });

        int32_t id = -1;

        if (found != m_triangles.end()) {
            const auto& tri = *found;
            id = tri->getId();
        }

        return id;
    }

    void context::addTransformable(const std::shared_ptr<aten::transformable>& t)
    {
        AT_ASSERT(t);
        m_transformables.push_back(t);
        t->updateIndex(m_transformables.size() - 1);
    }

    std::shared_ptr<const aten::transformable> context::getTransformable(int32_t idx) const
    {
        AT_ASSERT(0 <= idx && idx < m_transformables.size());
        return m_transformables[idx];
    }

    void context::traverseTransformables(
        std::function<void(const std::shared_ptr<aten::transformable>&, aten::ObjectType)> func) const
    {
        for (const auto& t : m_transformables) {
            auto type = t->getType();
            func(t, type);
        }
    }

    void context::copyMatricesAndUpdateTransformableMatrixIdx(std::vector<aten::mat4>& dst) const
    {
        traverseTransformables([&dst](const std::shared_ptr<aten::transformable>& t, aten::ObjectType type) {
            if (type == ObjectType::Instance) {
                aten::mat4 mtxL2W, mtxW2L;
                t->getMatrices(mtxL2W, mtxW2L);

                if (!mtxL2W.isIdentity()) {
                    auto& param = t->getParam();
                    param.mtx_id = (int32_t)(dst.size() / 2);

                    dst.push_back(mtxL2W);
                    dst.push_back(mtxW2L);
                }
            }
        });
    }

    int32_t context::findTransformableIdxFromPointer(const void* p) const
    {
        auto found = std::find_if(
            m_transformables.begin(), m_transformables.end(),
            [&](const std::shared_ptr<aten::transformable>& t) {
            return t.get() == p;
        });

        int32_t id = -1;

        if (found != m_transformables.end()) {
            const auto& t = *found;
            id = t->id();
        }

        return id;
    }

    int32_t context::findPolygonalTransformableOrderFromPointer(const void* p) const
    {
        int32_t order = -1;

        for (const auto& t : m_transformables) {
            auto type = t->getType();
            if (type == aten::ObjectType::Polygon) {
                order++;
            }

            if (t.get() == p) {
                break;
            }
        }

        AT_ASSERT(order >= 0);

        return order;
    }

    std::shared_ptr<texture> context::createTexture(
        uint32_t width, uint32_t height, uint32_t channels, const char* name)
    {
        auto ret = texture::create(width, height, channels, name);
        AT_ASSERT(ret);

        addTexture(ret);

        return ret;
    }

    int32_t context::getTextureNum() const
    {
        return m_textures.size();
    }

    std::shared_ptr<texture> context::getTexture(int32_t idx) const
    {
        AT_ASSERT(0 <= idx && idx < getTextureNum());
        return m_textures[idx];
    }

    void context::addTexture(const std::shared_ptr<texture>& tex)
    {
        AT_ASSERT(tex);
        m_textures.push_back(tex);
        tex->updateIndex(m_textures.size() - 1);
    }

    void context::initAllTexAsGLTexture()
    {
        auto num = getTextureNum();

        for (int32_t i = 0; i < num; i++) {
            auto tex = m_textures[i];
            tex->initAsGLTexture();
        }
    }
}
