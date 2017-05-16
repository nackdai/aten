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
	const aten::ray& ray,
	float t_min, float t_max,
	aten::hitrecord* rec)
{
	float4 p0 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[0] + 0);
	float4 p1 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[1] + 0);
	float4 p2 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[2] + 0);

	aten::vec3 v0 = aten::make_float3(p0.x, p0.y, p0.z);
	aten::vec3 v1 = aten::make_float3(p1.x, p1.y, p1.z);
	aten::vec3 v2 = aten::make_float3(p2.x, p2.y, p2.z);

	aten::vec3 e1 = v1 - v0;
	aten::vec3 e2 = v2 - v0;
	aten::vec3 r = ray.org - v0;
	aten::vec3 d = ray.dir;

	aten::vec3 u = cross(d, e2);
	aten::vec3 v = cross(r, e1);

	real inv = real(1) / dot(u, e1);

	real t = dot(v, e2) * inv;
	real beta = dot(u, r) * inv;
	real gamma = dot(v, d) * inv;

	bool isIntersect = ((beta >= real(0) && beta <= real(1))
		&& (gamma >= real(0) && gamma <= real(1))
		&& (beta + gamma <= real(1))
		&& t >= real(0));

	if (isIntersect) {
		if (t < rec->t) {
			rec->t = t;

			rec->param.a = beta;
			rec->param.b = gamma;

			rec->area = prim->area;

			rec->param.idx[0] = prim->idx[0];
			rec->param.idx[1] = prim->idx[1];
			rec->param.idx[2] = prim->idx[2];

			return true;
		}
	}

	return false;
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
	float4 p0 = tex1Dfetch<float4>(ctxt->vertices, 4 * rec->param.idx[0] + 0);
	float4 p1 = tex1Dfetch<float4>(ctxt->vertices, 4 * rec->param.idx[1] + 0);
	float4 p2 = tex1Dfetch<float4>(ctxt->vertices, 4 * rec->param.idx[2] + 0);

	float4 n0 = tex1Dfetch<float4>(ctxt->vertices, 4 * rec->param.idx[0] + 1);
	float4 n1 = tex1Dfetch<float4>(ctxt->vertices, 4 * rec->param.idx[1] + 1);
	float4 n2 = tex1Dfetch<float4>(ctxt->vertices, 4 * rec->param.idx[2] + 1);

	float4 u0 = tex1Dfetch<float4>(ctxt->vertices, 4 * rec->param.idx[0] + 2);
	float4 u1 = tex1Dfetch<float4>(ctxt->vertices, 4 * rec->param.idx[1] + 2);
	float4 u2 = tex1Dfetch<float4>(ctxt->vertices, 4 * rec->param.idx[2] + 2);

	//AT_NAME::face::evalHitResult(v0, v1, v2, rec);

	real a = rec->param.a;
	real b = rec->param.b;
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

	// tangent coordinate.
	rec->du = normalize(getOrthoVector(rec->normal));
	rec->dv = normalize(cross(rec->normal, rec->du));

	rec->p = param->mtxL2W.apply(rec->p);
	rec->normal = normalize(param->mtxL2W.applyXYZ(rec->normal));

	aten::vec3 v0 = aten::make_float3(p0.x, p0.y, p0.z);
	aten::vec3 v1 = aten::make_float3(p1.x, p1.y, p1.z);

	real orignalLen = (v1 - v0).length();

	real scaledLen = 0;
	{
		auto _p0 = param->mtxL2W.apply(v0);
		auto _p1 = param->mtxL2W.apply(v1);

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
	aten::hitrecord* rec)
{
	const aten::ShapeParameter* realShape = (param->shapeid >= 0 ? &ctxt->shapes[param->shapeid] : param);

	funcEvalHitResult[realShape->type](ctxt, param, r, rec);
}

__device__ void addIntersectFuncs()
{
	funcIntersect[aten::ShapeType::Polygon] = hitTriangle;
	funcIntersect[aten::ShapeType::Instance] = nullptr;
	funcIntersect[aten::ShapeType::Sphere] = hitSphere;
	funcIntersect[aten::ShapeType::Cube] = nullptr;

	funcEvalHitResult[aten::ShapeType::Polygon] = evalHitResultTriangle;
	funcEvalHitResult[aten::ShapeType::Instance] = nullptr;
	funcEvalHitResult[aten::ShapeType::Sphere] = evalHitResultSphere;
	funcEvalHitResult[aten::ShapeType::Cube] = nullptr;
}
