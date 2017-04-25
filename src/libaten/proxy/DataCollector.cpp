#include "proxy/DataCollector.h"
#include "shape/tranfomable.h"

namespace aten {
	void DataCollector::collect(
		std::vector<aten::ShapeParameter>& shapeparams,
		std::vector<aten::LightParameter>& lightparams,
		std::vector<aten::MaterialParameter>& mtrlparms)
	{
		const auto& shapes = aten::transformable::getShapes();

		for (auto s : shapes) {
			auto param = s->getParam();
			param.mtrl.idx = aten::material::findMaterialIdx((aten::material*)param.mtrl.ptr);
			shapeparams.push_back(param);
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