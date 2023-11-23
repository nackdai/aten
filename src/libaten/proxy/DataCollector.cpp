#include <algorithm>
#include <iterator>

#include "proxy/DataCollector.h"
#include "geometry/PolygonObject.h"

namespace aten {
    void DataCollector::collect(
        context& ctxt,
        std::vector<aten::ObjectParameter>& shapeparams,
        std::vector<aten::TriangleParameter>& primparams,
        std::vector<aten::LightParameter>& lightparams,
        std::vector<aten::MaterialParameter>& mtrlparms,
        std::vector<aten::vertex>& vtxparams,
        std::vector<aten::mat4>& mtxs)
    {
        mtxs = ctxt.PickNonIdentityMatricesAndUpdateMatrixIdxInTransformable();

        // Not guarantee order of the object which the instance has.
        ctxt.TraverseTransformables([&](const std::shared_ptr<aten::transformable>& s, aten::ObjectType type) {
            if (type == ObjectType::Instance) {
                auto param = s->GetParam();
                auto obj = s->getHasObject();

                auto idx = ctxt.FindTransformableIdxFromPointer(obj);

                // Specify the index of the object which the instance has.
                param.object_id = idx;

                // TODO
                param.area = ((aten::PolygonObject*)obj)->GetParam().area;

                shapeparams.push_back(param);
            }
            else if (type == ObjectType::Polygons) {
                auto param = s->GetParam();

                param.object_id = s->id();

                shapeparams.push_back(param);
            }
            else if (type == ObjectType::Sphere)
            {
                // TODO
                AT_ASSERT(false);
                auto param = s->GetParam();
                param.sphere.mtrl_id = -1;
                shapeparams.push_back(param);
            }
        });

        const auto lightNum = ctxt.GetLightNum();

        for (uint32_t i = 0; i < lightNum; i++) {
            const auto& param = ctxt.GetLight(i);
            lightparams.push_back(param);
        }

        ctxt.CopyMaterialParameters(mtrlparms);

        ctxt.CopyTriangleParameters(primparams);

        ctxt.CopyVertices(vtxparams);
    }

    void DataCollector::collectTriangles(
        const context& ctxt,
        std::vector<std::vector<aten::TriangleParameter>>& triangles,
        std::vector<int32_t>& triIdOffsets)
    {
        int32_t triangleCount = 0;

        ctxt.TraverseTransformables([&](const std::shared_ptr<aten::transformable>& s, aten::ObjectType type) {
            if (type == ObjectType::Polygons) {
                triangles.push_back(std::vector<aten::TriangleParameter>());
                auto pos = triangles.size() - 1;

                s->collectTriangles(triangles[pos]);

                triIdOffsets.push_back(triangleCount);
                triangleCount += static_cast<int32_t>(triangles[pos].size());
            }
        });
    }
}
