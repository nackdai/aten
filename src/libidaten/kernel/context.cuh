#pragma once

#include "aten4idaten.h"

struct Context {
	int geomnum;
	aten::ShapeParameter* shapes;

	aten::MaterialParameter* mtrls;

	int lightnum;
	aten::LightParameter* lights;

	cudaTextureObject_t* nodes;

	aten::PrimitiveParamter* prims;
	cudaTextureObject_t vertices;

	aten::mat4* matrices;
};
