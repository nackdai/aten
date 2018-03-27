#include "proxy/DataCollector.h"
#include "geometry/object.h"

#include <algorithm>
#include <iterator>

namespace aten {
	void DataCollector::collect(
		std::vector<aten::GeomParameter>& shapeparams,
		std::vector<aten::PrimitiveParamter>& primparams,
		std::vector<aten::LightParameter>& lightparams,
		std::vector<aten::MaterialParameter>& mtrlparms,
		std::vector<aten::vertex>& vtxparams)
	{
		const auto& shapes = aten::transformable::getShapes();
		std::vector<aten::transformable*> shapeOrderList;

		for (auto s : shapes) {
			shapeOrderList.push_back(s);
		}

		// Not gurantee order of the object which the instance has.
		for (auto s : shapes) {
			auto type = s->getParam().type;

			if (type == GeometryType::Instance) {
				auto param = s->getParam();
				auto obj = s->getHasObject();
				
				auto it = std::find(shapeOrderList.begin(), shapeOrderList.end(), (transformable*)(obj));
				AT_ASSERT(it != shapeOrderList.end());

				auto idx = std::distance(shapeOrderList.begin(), it);

				// Specify the index of the object which the instance has.
				param.shapeid = idx;

				// TODO
				param.area = ((aten::object*)obj)->getParam().area;

				shapeparams.push_back(param);
			}
			else if (type == GeometryType::Polygon) {
				auto param = s->getParam();
				shapeparams.push_back(param);
			}
			else if (type == GeometryType::Sphere
				|| type == GeometryType::Cube)
			{
				auto param = s->getParam();
				param.mtrl.idx = aten::material::findMaterialIdx((aten::material*)param.mtrl.ptr);
				shapeparams.push_back(param);
			}
		}

		const auto& lights = aten::Light::getLights();

		for (auto l : lights) {
			auto param = l->param();
			lightparams.push_back(param);
		}

		const auto& mtrls = aten::material::getMaterials();

		for (auto m : mtrls) {
			mtrlparms.push_back(m->param());
		}

		const auto& faces = aten::face::faces();

		for (auto f : faces) {
			primparams.push_back(f->param);
		}

		const auto& vtxs = aten::VertexManager::getVertices();
#if 0
		for (auto v : vtxs) {
			vtxparams.push_back(v);
		}
#else
		std::copy(
			vtxs.begin(),
			vtxs.end(),
			std::back_inserter(vtxparams));
#endif
	}

	void DataCollector::collectTriangles(
		std::vector<std::vector<aten::PrimitiveParamter>>& triangles,
		std::vector<int>& triIdOffsets,
		std::vector<aten::vertex>& vtxparams)
	{
		const auto& shapes = aten::transformable::getShapes();

		int triangleCount = 0;

		for (const auto s : shapes) {
			auto type = s->getParam().type;

			if (type == GeometryType::Polygon) {
				// TODO
				aten::object* obj = static_cast<aten::object*>(s);

				triangles.push_back(std::vector<aten::PrimitiveParamter>());
				int pos = triangles.size() - 1;

				for (const auto objshape : obj->shapes) {
					const auto& tris = objshape->tris();
					
					triangles[pos].reserve(tris.size());

					for (const auto tri : tris) {
						triangles[pos].push_back(tri->param);
					}
				}

				triIdOffsets.push_back(triangleCount);
				triangleCount += triangles[pos].size();
			}
		}

		const auto& vtxs = aten::VertexManager::getVertices();
		std::copy(
			vtxs.begin(),
			vtxs.end(),
			std::back_inserter(vtxparams));
	}
}