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

	cudaTextureObject_t vtxPos;
	cudaTextureObject_t vtxNml;

	aten::mat4* matrices;

	cudaTextureObject_t* textures;
	int envmapIdx;
};
