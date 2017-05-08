#include "proxy/DataCollector.h"
#include "shape/tranformable.h"

#include <map>

namespace aten {
	void DataCollector::collect(
		std::vector<aten::ShapeParameter>& shapeparams,
		std::vector<aten::PrimitiveParamter>& primparams,
		std::vector<aten::LightParameter>& lightparams,
		std::vector<aten::MaterialParameter>& mtrlparms)
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
				AT_ASSERT(param.primid == primparams.size());
				shapeparams.push_back(param);

				s->getPrimitives(primparams);

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

				param.shapeid = it->second;

				param.area = it->first->getParam().area;

				shapeparams.push_back(param);
			}
		}

		const auto& lights = aten::Light::getLights();

		for (auto l : lights) {
			auto param = l->param();
			param.object.idx = aten::transformable::findShapeIdx((aten::transformable*)param.object.ptr);
			lightparams.push_back(param);
		}

		const auto& mtrls = aten::material::getMaterials();

		for (auto m : mtrls) {
			mtrlparms.push_back(m->param());
		}
	}
}