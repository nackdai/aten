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
	aten::Intersection*);

__device__ FuncIntersect funcIntersect[aten::ShapeType::ShapeTypeMax];

__device__ bool hitSphere(
	const aten::ShapeParameter* shape,
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::Intersection* isect)
{
	return AT_NAME::sphere::hit(shape, r, t_min, t_max, isect);
}

__device__ bool hitTriangle(
	const aten::ShapeParameter* shape,
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& ray,
	float t_min, float t_max,
	aten::Intersection* isect)
{
	float4 p0 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[0] + 0);
	float4 p1 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[1] + 0);
	float4 p2 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[2] + 0);

	aten::vec3 v0 = aten::make_float3(p0.x, p0.y, p0.z);
	aten::vec3 v1 = aten::make_float3(p1.x, p1.y, p1.z);
	aten::vec3 v2 = aten::make_float3(p2.x, p2.y, p2.z);

#if 1
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
#else
	// NOTE
	// http://jcgt.org/published/0002/01/05/paper.pdf

	// calculate dimension where ray direction is maximal.
	int kz = aten::maxDim(ray.dir);
	int kx = (kz + 1) % 3;
	int ky = (kx + 1) % 3;

	// swap kx and ky dimension to preserve windin direction of triangles.
	if (ray.dir[kz] < real(0)) {
		int tmp = kx;
		kx = ky;
		ky = tmp;
	}

	// calculate shear constants.
	real Sx = ray.dir[kx] / ray.dir[kz];
	real Sy = ray.dir[ky] / ray.dir[kz];
	real Sz = real(1) / ray.dir[kz];

	// calculate vertices relative to ray origin.
	const auto A = v0 - ray.org;
	const auto B = v1 - ray.org;
	const auto C = v2 - ray.org;

	// perform shear and scale of vertices.
	const real Ax = A[kx] - Sx * A[kz];
	const real Ay = A[ky] - Sy * A[kz];
	const real Bx = B[kx] - Sx * B[kz];
	const real By = B[ky] - Sy * B[kz];
	const real Cx = C[kx] - Sx * C[kz];
	const real Cy = C[ky] - Sy * C[kz];

	// calculate scaled barycentric coordinates.
	real U = Cx * By - Cy * Bx;
	real V = Ax * Cy - Ay * Cx;
	real W = Bx * Ay - By * Ax;

	// Peform edge tests.
	// Moving this test before 
	// and the end of the previous conditional gives higher performance.
	if ((U < real(0) || V < real(0) || W < real(0))
		&& (U > real(0) || V > real(0) || W > real(0)))
	{
		return false;
	}

	// calculate dterminant.
	real det = U + V + W;

	if (det == real(0)) {
		return false;
	}

	// Calculate scaled z-coordinated of vertice
	// and use them to calculate the hit distance.
	const real Az = Sz * A[kz];
	const real Bz = Sz * B[kz];
	const real Cz = Sz * C[kz];
	const real T = U * Az + V * Bz + W * Cz;

	const real rcpDet = real(1) / det;

	const real beta = U * rcpDet;
	const real gamma = V * rcpDet;
	const real t = T * rcpDet;
#endif

	bool isIntersect = ((beta >= real(0) && beta <= real(1))
		&& (gamma >= real(0) && gamma <= real(1))
		&& (beta + gamma <= real(1))
		&& t >= real(0));

	if (isIntersect) {
		if (t < isect->t) {
			isect->t = t;

			isect->area = prim->area;

			isect->a = beta;
			isect->b = gamma;

			// NOTE
			// isect->primid value will be set later.

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
	aten::hitrecord* rec,
	aten::Intersection* isect)
{
	printf("Hit Test Not Supoorted[%d]\n", shape->type);
}

__device__ bool intersectShape(
	const aten::ShapeParameter* shape,
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::Intersection* isect)
{
	const aten::ShapeParameter* realShape = (shape->shapeid >= 0 ? &ctxt->shapes[shape->shapeid] : shape);

	auto ret = funcIntersect[realShape->type](realShape, prim, ctxt, r, t_min, t_max, isect);
	return ret;
}

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

	float4 p0 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[0] + 0);
	float4 p1 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[1] + 0);
	float4 p2 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[2] + 0);

	float4 n0 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[0] + 1);
	float4 n1 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[1] + 1);
	float4 n2 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[2] + 1);

	float4 u0 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[0] + 2);
	float4 u1 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[1] + 2);
	float4 u2 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[2] + 2);

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
	funcIntersect[aten::ShapeType::Polygon] = hitTriangle;
	funcIntersect[aten::ShapeType::Instance] = nullptr;
	funcIntersect[aten::ShapeType::Sphere] = hitSphere;
	funcIntersect[aten::ShapeType::Cube] = nullptr;

	funcEvalHitResult[aten::ShapeType::Polygon] = evalHitResultTriangle;
	funcEvalHitResult[aten::ShapeType::Instance] = nullptr;
	funcEvalHitResult[aten::ShapeType::Sphere] = evalHitResultSphere;
	funcEvalHitResult[aten::ShapeType::Cube] = nullptr;
}
