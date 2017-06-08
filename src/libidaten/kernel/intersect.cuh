#pragma once

#include "aten4idaten.h"
#include "kernel/context.cuh"

__device__ void addIntersectFuncs();

inline __device__ bool hitSphere(
	const aten::ShapeParameter* shape,
	const aten::ray& r,
	float t_min, float t_max,
	aten::Intersection* isect)
{
	return AT_NAME::sphere::hit(shape, r, t_min, t_max, isect);
}

inline __device__ bool hitTriangle(
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& ray,
	aten::Intersection* isect)
{
	float4 p0 = tex1Dfetch<float4>(ctxt->vtxPos, prim->idx[0]);
	float4 p1 = tex1Dfetch<float4>(ctxt->vtxPos, prim->idx[1]);
	float4 p2 = tex1Dfetch<float4>(ctxt->vtxPos, prim->idx[2]);

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

AT_DEVICE_API void evalHitResult(
	const Context* ctxt,
	const aten::ShapeParameter* param,
	const aten::ray& r,
	aten::hitrecord* rec,
	const aten::Intersection* isect);
