#include <algorithm>
#include <iterator>

#include "proxy/DataCollector.h"
#include "geometry/object.h"

namespace aten {
    void DataCollector::collect(
        const context& ctxt,
        const scene& scene,
        std::vector<aten::GeomParameter>& shapeparams,
        std::vector<aten::PrimitiveParamter>& primparams,
        std::vector<aten::LightParameter>& lightparams,
        std::vector<aten::MaterialParameter>& mtrlparms,
        std::vector<aten::vertex>& vtxparams)
    {
        const auto& shapes = aten::transformable::getShapes();

        // Not gurantee order of the object which the instance has.
        for (auto s : shapes) {
            auto type = s->getParam().type;

            if (type == GeometryType::Instance) {
                auto param = s->getParam();
                auto obj = s->getHasObject();
                
                auto idx = aten::transformable::findIdxAsHitable(obj);

                // Specify the index of the object which the instance has.
                param.shapeid = idx;

                // TODO
                param.area = ((aten::object*)obj)->getParam().area;

                shapeparams.push_back(param);
            }
            else if (type == GeometryType::Polygon) {
                auto param = s->getParam();

                auto idx = aten::transformable::findIdxAsHitable(s);
                param.shapeid = idx;

                shapeparams.push_back(param);
            }
            else if (type == GeometryType::Sphere
                || type == GeometryType::Cube)
            {
                auto param = s->getParam();
                auto mtrl = reinterpret_cast<aten::material*>(param.mtrl.ptr);
                param.mtrl.idx = mtrl->id();
                shapeparams.push_back(param);
            }
        }

        auto lightNum = scene.lightNum();

        for (uint32_t i = 0; i < lightNum; i++) {
            const auto& l = scene.getLight(i);
            auto param = l->param();
            lightparams.push_back(param);
        }

        ctxt.copyMaterialParameters(mtrlparms);

        const auto& faces = aten::face::faces();

        for (auto f : faces) {
            primparams.push_back(f->getParam());
        }

        ctxt.copyVertices(vtxparams);
    }

    void DataCollector::collectTriangles(
        std::vector<std::vector<aten::PrimitiveParamter>>& triangles,
        std::vector<int>& triIdOffsets)
    {
        const auto& shapes = aten::transformable::getShapesPolygonObjList();

        int triangleCount = 0;

        for (const auto s : shapes) {
            auto type = s->getParam().type;

            if (type == GeometryType::Polygon) {
                triangles.push_back(std::vector<aten::PrimitiveParamter>());
                int pos = triangles.size() - 1;

                s->collectTriangles(triangles[pos]);

                triIdOffsets.push_back(triangleCount);
                triangleCount += triangles[pos].size();
            }
        }
    }
}