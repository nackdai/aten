#include "shape/tranformable.h"

namespace aten
{
	std::vector<transformable*> transformable::g_shapes;

	transformable::transformable()
	{
		m_id = g_shapes.size();
		g_shapes.push_back(this);
	}

	transformable::~transformable()
	{
		auto it = std::find(g_shapes.begin(), g_shapes.end(), this);
		if (it != g_shapes.end()) {
			g_shapes.erase(it);
		}
	}

	uint32_t transformable::getShapeNum()
	{
		return (uint32_t)g_shapes.size();
	}

	transformable* transformable::getShape(uint32_t idx)
	{
		if (idx < g_shapes.size()) {
			return g_shapes[idx];
		}
		return nullptr;
	}

	transformable* transformable::getShapeAsHitable(const hitable* shape)
	{
		transformable* ret = nullptr;

		auto found = std::find(g_shapes.begin(), g_shapes.end(), shape);
		if (found != g_shapes.end()) {
			ret = *found;
		}

		return ret;
	}

	int transformable::findShapeIdx(const transformable* shape)
	{
		auto found = std::find(g_shapes.begin(), g_shapes.end(), shape);
		if (found != g_shapes.end()) {
			auto id = std::distance(g_shapes.begin(), found);
			AT_ASSERT(shape == g_shapes[id]);
			return id;
		}
		return -1;
	}

	int transformable::findShapeIdxAsHitable(const hitable* shape)
	{
		if (shape) {
			auto found = std::find(g_shapes.begin(), g_shapes.end(), shape);
			if (found != g_shapes.end()) {
				auto id = std::distance(g_shapes.begin(), found);
				AT_ASSERT(shape == g_shapes[id]);
				return id;
			}
		}
		return -1;
	}

	const std::vector<transformable*>& transformable::getShapes()
	{
		return g_shapes;
	}

	void transformable::gatherAllTransformMatrixAndSetMtxIdx(std::vector<aten::mat4>& mtxs)
	{
		auto& shapes = const_cast<std::vector<transformable*>&>(transformable::getShapes());

		for (auto s : shapes) {
			auto& param = const_cast<aten::ShapeParameter&>(s->getParam());

			if (param.type == ShapeType::Instance) {
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
