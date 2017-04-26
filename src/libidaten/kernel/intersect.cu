#include "kernel/intersect.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda/helper_math.h"

typedef bool(*FuncIntersect)(const aten::ShapeParameter&, const aten::ray&, float t_min, float t_max, aten::hitrecord&);

__device__ FuncIntersect funcIntersect[aten::ShapeType::ShapeTypeMax];

__device__ void addIntersectFuncs()
{
	funcIntersect[aten::ShapeType::Object] = nullptr;
	funcIntersect[aten::ShapeType::Mesh] = nullptr;
	funcIntersect[aten::ShapeType::Sphere] = AT_NAME::sphere::hit;
	funcIntersect[aten::ShapeType::Cube] = nullptr;
}

__device__ bool intersectShape(
	const aten::ShapeParameter& shape,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord& rec)
{
	auto ret = funcIntersect[shape.type](shape, r, t_min, t_max, rec);
	return ret;
}
