#pragma once

#include "aten4idaten.h"

struct Context {
	int geomnum;
	aten::ShapeParameter* shapes;

	aten::MaterialParameter* mtrls;

	int lightnum;
	aten::LightParameter* lights;

	aten::BVHNode* nodes;
};
