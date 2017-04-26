#include "shape/tranformable.h"

namespace aten
{
	std::vector<transformable*> transformable::g_shapes;

	transformable::transformable()
	{
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

	const transformable* transformable::getShape(uint32_t idx)
	{
		if (idx < g_shapes.size()) {
			return g_shapes[idx];
		}
		return nullptr;
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
		auto found = std::find(g_shapes.begin(), g_shapes.end(), shape);
		if (found != g_shapes.end()) {
			auto id = std::distance(g_shapes.begin(), found);
			AT_ASSERT(shape == g_shapes[id]);
			return id;
		}
		return -1;
	}

	const std::vector<transformable*>& transformable::getShapes()
	{
		return g_shapes;
	}
}
