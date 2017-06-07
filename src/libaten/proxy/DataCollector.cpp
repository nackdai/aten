#include "proxy/DataCollector.h"
#include "shape/tranformable.h"
#include "object/object.h"

#include <map>

namespace aten {
	void DataCollector::collect(
		std::vector<aten::ShapeParameter>& shapeparams,
		std::vector<aten::PrimitiveParamter>& primparams,
		std::vector<aten::LightParameter>& lightparams,
		std::vector<aten::MaterialParameter>& mtrlparms,
		std::vector<aten::vertex>& vtxparams)
	{
		const auto& shapes = aten::transformable::getShapes();

		int order = 0;
		std::map<transformable*, int> orderMap;

		for (auto s : shapes) {
			auto type = s->getParam().type;

			switch (type) {
			case ShapeType::Polygon:
			{
				auto param = s->getParam();
				shapeparams.push_back(param);

				const int primcnt = primparams.size();

				if (primcnt <= param.primid) {
					primparams.resize(primparams.size() + param.primid + param.primnum);
				}

				s->getPrimitives(&primparams[param.primid]);

				orderMap.insert(std::pair<transformable*, int>(s, order++));
			}
				break;
			case ShapeType::Sphere:
			case ShapeType::Cube:
			{
				auto param = s->getParam();
				param.mtrl.idx = aten::material::findMaterialIdx((aten::material*)param.mtrl.ptr);
				shapeparams.push_back(param);

				orderMap.insert(std::pair<transformable*, int>(s, order++));
			}
				break;
			default:
				break;
			}	
		}

		// Not gurantee order of the object which the instance has.
		for (auto s : shapes) {
			auto type = s->getParam().type;

			if (type == ShapeType::Instance) {
				auto param = s->getParam();
				auto obj = s->getHasObject();
				
				auto it = orderMap.find((transformable*)(obj));
				AT_ASSERT(it != orderMap.end());

				// Specify the index of the object which the instance has.
				param.shapeid = it->second;

				param.area = it->first->getParam().area;

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

		for (auto v : vtxs) {
			vtxparams.push_back(v);
		}
	}
}