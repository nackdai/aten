#include "proxy/DataCollector.h"
#include "shape/tranfomable.h"

namespace aten {
	void DataCollector::collect(
		std::vector<aten::ShapeParameter>& shapeparams,
		std::vector<aten::PrimitiveParamter>& primparams,
		std::vector<aten::LightParameter>& lightparams,
		std::vector<aten::MaterialParameter>& mtrlparms)
	{
		const auto& shapes = aten::transformable::getShapes();

		for (auto s : shapes) {
			auto type = s->getParam().type;

			switch (type) {
			case ShapeType::Mesh:
				s->getShapes(shapeparams, primparams);
				break;
			case ShapeType::Sphere:
			case ShapeType::Cube:
			{
				auto param = s->getParam();
				param.mtrl.idx = aten::material::findMaterialIdx((aten::material*)param.mtrl.ptr);
				shapeparams.push_back(param);
			}
				break;
			default:
				break;
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