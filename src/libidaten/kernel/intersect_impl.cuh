#include "kernel/idatendefs.cuh"

AT_CUDA_INLINE __device__ bool hitSphere(
	const aten::ShapeParameter* shape,
	const aten::ray& r,
	float t_min, float t_max,
	aten::Intersection* isect)
{
	return AT_NAME::sphere::hit(shape, r, t_min, t_max, isect);
}

AT_CUDA_INLINE __device__ bool hitTriangle(
	const aten::PrimitiveParamter* prim,
	const Context* ctxt,
	const aten::ray& ray,
	aten::Intersection* isect)
{
	float4 p0 = tex1Dfetch<float4>(ctxt->vtxPos, prim->idx[0]);
	float4 p1 = tex1Dfetch<float4>(ctxt->vtxPos, prim->idx[1]);
	float4 p2 = tex1Dfetch<float4>(ctxt->vtxPos, prim->idx[2]);

	aten::vec3 v0 = aten::vec3(p0.x, p0.y, p0.z);
	aten::vec3 v1 = aten::vec3(p1.x, p1.y, p1.z);
	aten::vec3 v2 = aten::vec3(p2.x, p2.y, p2.z);

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

	const real alpha = real(1) - beta - gamma;

	bool isIntersect = (beta >= real(0))
		&& (gamma >= real(0))
		&& (alpha >= real(0))
		&& (t >= real(0));

	if (isIntersect) {
		isect->t = t;

		isect->area = prim->area;

		isect->a = beta;
		isect->b = gamma;

		// NOTE
		// isect->primid value will be set later.
	}

	return isIntersect;
}

inline __device__ float4 vmax(float4 a, float4 b)
{
	return make_float4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

inline __device__ float4 vmin(float4 a, float4 b)
{
	return make_float4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

inline __device__ float max3(float a, float b, float c)
{
	return max(max(a, b), c);
}

inline __device__ float min3(float a, float b, float c)
{
	return min(min(a, b), c);
}

inline __device__ float2 _hitAABB(
	aten::vec3 org,
	aten::vec3 dir,
	float4 boxmin, float4 boxmax,
	real t_min, real t_max)
{
	float4 invdir = make_float4(1.0f / (dir.x + 1e-6f), 1.0f / (dir.y + 1e-6f), 1.0f / (dir.z + 1e-6f), 1.0f);

	float4 oxinvdir = make_float4(org.x, org.y, org.z, 1.0f);
	oxinvdir = -oxinvdir * invdir;

	const float4 f = boxmax * invdir + oxinvdir;
	const float4 n = boxmin * invdir + oxinvdir;

	const float4 tmax = vmax(f, n);
	const float4 tmin = vmin(f, n);

	const float t1 = min(min3(tmax.x, tmax.y, tmax.z), t_max);
	const float t0 = max(max3(tmin.x, tmin.y, tmin.z), t_min);

	return make_float2(t0, t1);
}

AT_CUDA_INLINE __device__ bool hitAABB(
	aten::vec3 org,
	aten::vec3 dir,
	float4 boxmin, float4 boxmax,
	real t_min, real t_max)
{
	auto s = _hitAABB(org, dir, boxmin, boxmax, t_min, t_max);

	return s.x <= s.y;
}

AT_CUDA_INLINE __device__ bool hitAABB(
	aten::vec3 org,
	aten::vec3 dir,
	float4 boxmin, float4 boxmax,
	real t_min, real t_max,
	real* t_result)
{
	auto s = _hitAABB(org, dir, boxmin, boxmax, t_min, t_max);

	*t_result = s.x;

	return s.x <= s.y;
}

inline __device__ float4 cross(float4 a, float4 b)
{
	return make_float4(
		a.y * b.z - a.z * b.y, 
		a.z * b.x - a.x * b.z, 
		a.x * b.y - a.y * b.x,
		0);
}

AT_CUDA_INLINE __device__ void evalHitResultTriangle(
	const Context* ctxt,
	const aten::ShapeParameter* param,
	const aten::ray& r,
	aten::hitrecord* rec,
	const aten::Intersection* isect)
{
	int primidx = isect->primid;
	aten::PrimitiveParamter prim;
	prim.v0 = ((aten::vec4*)ctxt->prims)[primidx * aten::PrimitiveParamter_float4_size + 0];
	prim.v1 = ((aten::vec4*)ctxt->prims)[primidx * aten::PrimitiveParamter_float4_size + 1];

	float4 p0 = tex1Dfetch<float4>(ctxt->vtxPos, prim.idx[0]);
	float4 p1 = tex1Dfetch<float4>(ctxt->vtxPos, prim.idx[1]);
	float4 p2 = tex1Dfetch<float4>(ctxt->vtxPos, prim.idx[2]);

	float4 n0 = tex1Dfetch<float4>(ctxt->vtxNml, prim.idx[0]);
	float4 n1 = tex1Dfetch<float4>(ctxt->vtxNml, prim.idx[1]);
	float4 n2 = tex1Dfetch<float4>(ctxt->vtxNml, prim.idx[2]);

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

	if (prim.needNormal > 0) {
		float4 e01 = p1 - p0;
		float4 e02 = p2 - p0;

		n = normalize(cross(e01, e02));
	}

	rec->p = aten::vec3(p.x, p.y, p.z);
	rec->normal = aten::vec3(n.x, n.y, n.z);

	rec->u = uv.x;
	rec->v = uv.y;

	aten::vec3 v0 = aten::vec3(p0.x, p0.y, p0.z);
	aten::vec3 v1 = aten::vec3(p1.x, p1.y, p1.z);

	real orignalLen = (v1 - v0).length();

	real scaledLen = 0;

	if (param->mtxid >= 0) {
		auto mtxL2W = ctxt->matrices[param->mtxid * 2 + 0];

		rec->p = mtxL2W.apply(rec->p);
		rec->normal = normalize(mtxL2W.applyXYZ(rec->normal));

		{
			auto _p0 = mtxL2W.apply(v0);
			auto _p1 = mtxL2W.apply(v1);

			scaledLen = (_p1 - _p0).length();
		}
	}
	else {
		scaledLen = (v1 - v0).length();
	}

	real ratio = scaledLen / orignalLen;
	ratio = ratio * ratio;

	rec->area = param->area * ratio;
}

AT_CUDA_INLINE __device__ void evalHitResult(
	const Context* ctxt,
	const aten::ShapeParameter* param,
	const aten::ray& r,
	aten::hitrecord* rec,
	const aten::Intersection* isect)
{
	const aten::ShapeParameter* realShape = (param->shapeid >= 0 ? &ctxt->shapes[param->shapeid] : param);

	if (realShape->type == aten::ShapeType::Polygon) {
		evalHitResultTriangle(ctxt, param, r, rec, isect);
	}
	else if (realShape->type == aten::ShapeType::Sphere) {
		AT_NAME::sphere::evalHitResult(param, r, rec, isect);
	}
	else {
		// TODO
	}

	rec->objid = isect->objid;
	rec->mtrlid = isect->mtrlid;

#ifdef ENABLE_TANGENTCOORD_IN_HITREC
	// tangent coordinate.
	rec->du = normalize(getOrthoVector(rec->normal));
	rec->dv = normalize(cross(rec->normal, rec->du));
#endif
}
