#include <algorithm>
#include <iterator>

#include "proxy/DataCollector.h"
#include "geometry/object.h"

namespace aten {
    void DataCollector::collect(
        const context& ctxt,
        const scene& scene,
        std::vector<aten::ObjectParameter>& shapeparams,
        std::vector<aten::TriangleParameter>& primparams,
        std::vector<aten::LightParameter>& lightparams,
        std::vector<aten::MaterialParameter>& mtrlparms,
        std::vector<aten::vertex>& vtxparams)
    {
        // Not guarantee order of the object which the instance has.
        ctxt.traverseTransformables([&](const std::shared_ptr<aten::transformable>& s, aten::ObjectType type) {
            if (type == ObjectType::Instance) {
                auto param = s->getParam();
                auto obj = s->getHasObject();

                auto idx = ctxt.findTransformableIdxFromPointer(obj);

                // Specify the index of the object which the instance has.
                param.object_id = idx;

                // TODO
                param.area = ((aten::object*)obj)->getParam().area;

                shapeparams.push_back(param);
            }
            else if (type == ObjectType::Polygon) {
                auto param = s->getParam();

                param.object_id = s->id();

                shapeparams.push_back(param);
            }
            else if (type == ObjectType::Sphere)
            {
                auto param = s->getParam();
                param.sphere.mtrl_id = s->getMaterial()->id();
                shapeparams.push_back(param);
            }
        });

        auto lightNum = scene.lightNum();

        for (uint32_t i = 0; i < lightNum; i++) {
            auto l = scene.getLight(i);
            auto param = l->param();
            lightparams.push_back(param);
        }

        ctxt.copyMaterialParameters(mtrlparms);

        ctxt.copyPrimitiveParameters(primparams);

        ctxt.copyVertices(vtxparams);
    }

    void DataCollector::collectTriangles(
        const context& ctxt,
        std::vector<std::vector<aten::TriangleParameter>>& triangles,
        std::vector<int32_t>& triIdOffsets)
    {
        int32_t triangleCount = 0;

        ctxt.traverseTransformables([&](const std::shared_ptr<aten::transformable>& s, aten::ObjectType type) {
            if (type == ObjectType::Polygon) {
                triangles.push_back(std::vector<aten::TriangleParameter>());
                auto pos = triangles.size() - 1;

                s->collectTriangles(triangles[pos]);

                triIdOffsets.push_back(triangleCount);
                triangleCount += triangles[pos].size();
            }
        });
    }
}
