#include "geometry/transformable.h"

namespace aten
{
    std::vector<transformable*> transformable::g_transformables;
    std::vector<transformable*> transformable::g_transformablesPolygonObjList;

    transformable::transformable()
    {
        m_id = g_transformables.size();
        g_transformables.push_back(this);
    }

    transformable::transformable(GeometryType type)
    {
        m_id = g_transformables.size();
        g_transformables.push_back(this);

        if (type == GeometryType::Polygon) {
            g_transformablesPolygonObjList.push_back(this);
        }
    }

    transformable::~transformable()
    {
        {
            auto it = std::find(g_transformables.begin(), g_transformables.end(), this);
            if (it != g_transformables.end()) {
                g_transformables.erase(it);

                // IDÇÃêUÇËíºÇµ...
                for (int i = 0; i < g_transformables.size(); i++) {
                    g_transformables[i]->m_id = i;
                }
            }
        }

        auto it = std::find(g_transformablesPolygonObjList.begin(), g_transformablesPolygonObjList.end(), this);
        if (it != g_transformablesPolygonObjList.end()) {
            g_transformablesPolygonObjList.erase(it);
        }
    }

    uint32_t transformable::getShapeNum()
    {
        return (uint32_t)g_transformables.size();
    }

    transformable* transformable::getShape(uint32_t idx)
    {
        if (idx < g_transformables.size()) {
            return g_transformables[idx];
        }
        return nullptr;
    }

    transformable* transformable::getShapeAsHitable(const hitable* shape)
    {
        transformable* ret = nullptr;

        auto found = std::find(g_transformables.begin(), g_transformables.end(), shape);
        if (found != g_transformables.end()) {
            ret = *found;
        }

        return ret;
    }

    int transformable::findIdx(const transformable* shape)
    {
        return findIdxAsHitable((const hitable*)shape);
    }

    int transformable::findIdxAsHitable(const hitable* shape)
    {
        if (shape) {
            auto found = std::find(g_transformables.begin(), g_transformables.end(), shape);
            if (found != g_transformables.end()) {
                auto id = std::distance(g_transformables.begin(), found);
                AT_ASSERT(shape == g_transformables[id]);
                return id;
            }
        }
        return -1;
    }

    int transformable::findIdxFromPolygonObjList(const hitable* shape)
    {
        if (shape) {
            auto found = std::find(g_transformablesPolygonObjList.begin(), g_transformablesPolygonObjList.end(), shape);
            if (found != g_transformablesPolygonObjList.end()) {
                auto id = std::distance(g_transformablesPolygonObjList.begin(), found);
                AT_ASSERT(shape == g_transformablesPolygonObjList[id]);
                return id;
            }
        }
        return -1;
    }

    const std::vector<transformable*>& transformable::getShapes()
    {
        return g_transformables;
    }

    const std::vector<transformable*>& transformable::getShapesPolygonObjList()
    {
        return g_transformablesPolygonObjList;
    }

    void transformable::gatherAllTransformMatrixAndSetMtxIdx(std::vector<aten::mat4>& mtxs)
    {
        auto& shapes = const_cast<std::vector<transformable*>&>(transformable::getShapes());

        for (auto s : shapes) {
            auto& param = const_cast<aten::GeomParameter&>(s->getParam());

            if (param.type == GeometryType::Instance) {
                aten::mat4 mtxL2W, mtxW2L;
                s->getMatrices(mtxL2W, mtxW2L);

                if (!mtxL2W.isIdentity()) {
                    param.mtxid = (int)(mtxs.size() / 2);

                    mtxs.push_back(mtxL2W);
                    mtxs.push_back(mtxW2L);
                }
            }
        }
    }
}
