#include "kernel/intersect.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda/helper_math.h"
#include "aten4idaten.h"

typedef bool(*FuncIntersect)(const aten::ShapeParameter&, const aten::PrimitiveParamter*, const Context*, const aten::ray&, float, float, aten::hitrecord&);

__device__ FuncIntersect funcIntersect[aten::ShapeType::ShapeTypeMax];

__device__ bool hitSphere(
	const aten::ShapeParameter& shape,
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord& rec)
{
	return AT_NAME::sphere::hit(shape, r, t_min, t_max, rec);
}

__device__ bool hitTriangle(
	const aten::ShapeParameter& shape,
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord& rec)
{
	const auto& v0 = ctxt->vertices[prim->idx[0]];
	const auto& v1 = ctxt->vertices[prim->idx[1]];
	const auto& v2 = ctxt->vertices[prim->idx[2]];

	bool isHit = idaten::face::hit(
		*prim,
		v0, v1, v2,
		r,
		t_min, t_max,
		rec);

	return isHit;
}

__device__ bool hitNotSupported(
	const aten::ShapeParameter& shape,
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord& rec)
{
	printf("Hit Test Not Supoorted[%d]\n", shape.type);
}


__device__ void addIntersectFuncs()
{
	funcIntersect[aten::ShapeType::Polygon] = hitTriangle;
	funcIntersect[aten::ShapeType::Instance] = hitNotSupported;
	funcIntersect[aten::ShapeType::Sphere] = hitSphere;
	funcIntersect[aten::ShapeType::Cube] = hitNotSupported;
}

AT_DEVICE_API bool intersectShape(
	const aten::ShapeParameter& shape,
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord& rec)
{
	const aten::ShapeParameter* realShape = (shape.shapeid >= 0 ? &ctxt->shapes[shape.shapeid] : &shape);

	auto ret = funcIntersect[realShape->type](*realShape, prim, ctxt, r, t_min, t_max, rec);
	return ret;
}
