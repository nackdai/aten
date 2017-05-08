#include "kernel/intersect.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda/helper_math.h"

typedef bool(*FuncIntersect)(const aten::ShapeParameter&, const aten::PrimitiveParamter*, const Context&, const aten::ray&, float, float, aten::hitrecord&);

__device__ FuncIntersect funcIntersect[aten::ShapeType::ShapeTypeMax];

__device__ bool hitSphere(
	const aten::ShapeParameter& shape,
	const aten::PrimitiveParamter* prim,
	const Context& ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord& rec)
{
	return AT_NAME::sphere::hit(shape, r, t_min, t_max, rec);
}

__device__ void addIntersectFuncs()
{
	funcIntersect[aten::ShapeType::Polygon] = nullptr;
	funcIntersect[aten::ShapeType::Instance] = nullptr;
	funcIntersect[aten::ShapeType::Sphere] = hitSphere;
	funcIntersect[aten::ShapeType::Cube] = nullptr;
}

AT_DEVICE_API bool intersectShape(
	const aten::ShapeParameter& shape,
	const aten::PrimitiveParamter* prim,
	const Context& ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord& rec)
{
	auto ret = funcIntersect[shape.type](shape, prim, ctxt, r, t_min, t_max, rec);
	return ret;
}
