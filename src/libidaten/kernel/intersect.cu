#include "kernel/intersect.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "aten4idaten.h"

typedef bool(*FuncIntersect)(
	const aten::ShapeParameter*, 
	const aten::PrimitiveParamter*, 
	const Context*, 
	const aten::ray&, 
	float, 
	float, 
	aten::hitrecord*);

__device__ FuncIntersect funcIntersect[aten::ShapeType::ShapeTypeMax];

__device__ bool hitSphere(
	const aten::ShapeParameter* shape,
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord* rec)
{
	return AT_NAME::sphere::hit(shape, r, t_min, t_max, rec);
}

__device__ bool hitTriangle(
	const aten::ShapeParameter* shape,
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord* rec)
{
	const auto& v0 = ctxt->vertices[prim->idx[0]];
	const auto& v1 = ctxt->vertices[prim->idx[1]];
	const auto& v2 = ctxt->vertices[prim->idx[2]];

	bool isHit = idaten::face::hit(
		prim,
		v0, v1, v2,
		r,
		t_min, t_max,
		rec);

	return isHit;
}

__device__ bool hitNotSupported(
	const aten::ShapeParameter* shape,
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord* rec)
{
	printf("Hit Test Not Supoorted[%d]\n", shape->type);
}

AT_DEVICE_API bool intersectShape(
	const aten::ShapeParameter* shape,
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord* rec)
{
	const aten::ShapeParameter* realShape = (shape->shapeid >= 0 ? &ctxt->shapes[shape->shapeid] : shape);

	auto ret = funcIntersect[realShape->type](realShape, prim, ctxt, r, t_min, t_max, rec);
	return ret;
}

typedef void(*FuncEvalHitResult)(const Context*, const aten::ShapeParameter*, const aten::ray&, aten::hitrecord*);

__device__ FuncEvalHitResult funcEvalHitResult[aten::ShapeType::ShapeTypeMax];

__device__ void funcEvalHitResultNotSupported(
	const Context* ctxt,
	const aten::ShapeParameter* param,
	const aten::ray& r,
	aten::hitrecord* rec)
{
	printf("Eval Hit Result Not Supoorted[%d]\n", param->type);
}

__device__ void evalHitResultSphere(
	const Context* ctxt,
	const aten::ShapeParameter* param,
	const aten::ray& r,
	aten::hitrecord* rec)
{
	AT_NAME::sphere::evalHitResult(param, r, rec);
}

__device__ void evalHitResultTriangle(
	const Context* ctxt,
	const aten::ShapeParameter* param,
	const aten::ray& r,
	aten::hitrecord* rec)
{
	const auto& v0 = ctxt->vertices[rec->param.idx[0]];
	const auto& v1 = ctxt->vertices[rec->param.idx[1]];
	const auto& v2 = ctxt->vertices[rec->param.idx[2]];

	AT_NAME::face::evalHitResult(v0, v1, v2, rec);

	rec->p = param->mtxL2W.apply(rec->p);
	rec->normal = normalize(param->mtxL2W.applyXYZ(rec->normal));

	real orignalLen = (v1.pos - v0.pos).length();

	real scaledLen = 0;
	{
		auto p0 = param->mtxL2W.apply(v0.pos);
		auto p1 = param->mtxL2W.apply(v1.pos);

		scaledLen = (p1 - p0).length();
	}

	real ratio = scaledLen / orignalLen;
	ratio = ratio * ratio;

	rec->area = param->area * ratio;
}

AT_DEVICE_API void evalHitResult(
	const Context* ctxt,
	const aten::ShapeParameter* param,
	const aten::ray& r,
	aten::hitrecord* rec)
{
	const aten::ShapeParameter* realShape = (param->shapeid >= 0 ? &ctxt->shapes[param->shapeid] : param);

	funcEvalHitResult[realShape->type](ctxt, param, r, rec);

}

__device__ void addIntersectFuncs()
{
	funcIntersect[aten::ShapeType::Polygon] = hitTriangle;
	funcIntersect[aten::ShapeType::Instance] = hitNotSupported;
	funcIntersect[aten::ShapeType::Sphere] = hitSphere;
	funcIntersect[aten::ShapeType::Cube] = hitNotSupported;

	funcEvalHitResult[aten::ShapeType::Polygon] = evalHitResultTriangle;
	funcEvalHitResult[aten::ShapeType::Instance] = funcEvalHitResultNotSupported;
	funcEvalHitResult[aten::ShapeType::Sphere] = evalHitResultSphere;
	funcEvalHitResult[aten::ShapeType::Cube] = funcEvalHitResultNotSupported;
}
