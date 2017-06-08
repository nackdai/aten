#include "kernel/intersect.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "aten4idaten.h"

typedef void(*FuncEvalHitResult)(
	const Context*, 
	const aten::ShapeParameter*, 
	const aten::ray&, 
	aten::hitrecord*,
	const aten::Intersection*);

__device__ FuncEvalHitResult funcEvalHitResult[aten::ShapeType::ShapeTypeMax];

__device__ void funcEvalHitResultNotSupported(
	const Context* ctxt,
	const aten::ShapeParameter* param,
	const aten::ray& r,
	aten::hitrecord* rec,
	const aten::Intersection* isect)
{
	printf("Eval Hit Result Not Supoorted[%d]\n", param->type);
}

__device__ void evalHitResultSphere(
	const Context* ctxt,
	const aten::ShapeParameter* param,
	const aten::ray& r,
	aten::hitrecord* rec,
	const aten::Intersection* isect)
{
	AT_NAME::sphere::evalHitResult(param, r, rec, isect);
}

__device__ void evalHitResultTriangle(
	const Context* ctxt,
	const aten::ShapeParameter* param,
	const aten::ray& r,
	aten::hitrecord* rec,
	const aten::Intersection* isect)
{
	auto prim = &ctxt->prims[isect->primid];

	float4 p0 = tex1Dfetch<float4>(ctxt->vtxPos, prim->idx[0]);
	float4 p1 = tex1Dfetch<float4>(ctxt->vtxPos, prim->idx[1]);
	float4 p2 = tex1Dfetch<float4>(ctxt->vtxPos, prim->idx[2]);

	float4 n0 = tex1Dfetch<float4>(ctxt->vtxNml, prim->idx[0]);
	float4 n1 = tex1Dfetch<float4>(ctxt->vtxNml, prim->idx[1]);
	float4 n2 = tex1Dfetch<float4>(ctxt->vtxNml, prim->idx[2]);

	float2 u0 = make_float2(p0.w, n0.w);
	float2 u1 = make_float2(p1.w, n1.w);
	float2 u2 = make_float2(p2.w, n2.w);

	//AT_NAME::face::evalHitResult(v0, v1, v2, rec);

	real a = isect->a;
	real b = isect->b;
	real c = 1 - a - b;

	// dSÀ•WŒn(barycentric coordinates).
	// v0Šî€.
	// p = (1 - a - b)*v0 + a*v1 + b*v2
	auto p = c * p0 + a * p1 + b * p2;
	auto n = c * n0 + a * n1 + b * n2;
	auto uv = c * u0 + a * u1 + b * u2;

	rec->p = aten::make_float3(p.x, p.y, p.z);
	rec->normal = aten::make_float3(n.x, n.y, n.z);

	rec->u = uv.x;
	rec->v = uv.y;

	auto mtxL2W = ctxt->matrices[param->mtxid * 2 + 0];

	rec->p = mtxL2W.apply(rec->p);
	rec->normal = normalize(mtxL2W.applyXYZ(rec->normal));

	aten::vec3 v0 = aten::make_float3(p0.x, p0.y, p0.z);
	aten::vec3 v1 = aten::make_float3(p1.x, p1.y, p1.z);

	real orignalLen = (v1 - v0).length();

	real scaledLen = 0;
	{
		auto _p0 = mtxL2W.apply(v0);
		auto _p1 = mtxL2W.apply(v1);

		scaledLen = (_p1 - _p0).length();
	}

	real ratio = scaledLen / orignalLen;
	ratio = ratio * ratio;

	rec->area = param->area * ratio;
}

AT_DEVICE_API void evalHitResult(
	const Context* ctxt,
	const aten::ShapeParameter* param,
	const aten::ray& r,
	aten::hitrecord* rec,
	const aten::Intersection* isect)
{
	const aten::ShapeParameter* realShape = (param->shapeid >= 0 ? &ctxt->shapes[param->shapeid] : param);

	funcEvalHitResult[realShape->type](ctxt, param, r, rec, isect);

	rec->objid = isect->objid;
	rec->mtrlid = isect->mtrlid;

#ifdef ENABLE_TANGENTCOORD_IN_HITREC
	// tangent coordinate.
	rec->du = normalize(getOrthoVector(rec->normal));
	rec->dv = normalize(cross(rec->normal, rec->du));
#endif
}

__device__ void addIntersectFuncs()
{
	funcEvalHitResult[aten::ShapeType::Polygon] = evalHitResultTriangle;
	funcEvalHitResult[aten::ShapeType::Instance] = nullptr;
	funcEvalHitResult[aten::ShapeType::Sphere] = evalHitResultSphere;
	funcEvalHitResult[aten::ShapeType::Cube] = nullptr;
}
