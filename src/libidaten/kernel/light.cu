#include "kernel/light.cuh"
#include "kernel/intersect.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"

#include "aten4idaten.h"

typedef void(*FuncLightSample)(
	aten::LightSampleResult*, Context*, const aten::LightParameter*, const aten::vec3&, aten::sampler*);

typedef void(*FuncGetSamplePosNormalArea)(
	aten::hitable::SamplePosNormalPdfResult*, Context*, aten::ShapeParameter*, aten::sampler*);

__device__ void getSphereSamplePosNormalArea(
	aten::hitable::SamplePosNormalPdfResult* result,
	Context* ctxt, 
	aten::ShapeParameter* shape,
	aten::sampler* smpl)
{
	AT_NAME::sphere::getSamplePosNormalArea(result, shape, smpl);
}

__device__ void getSamplePosNormalAreaNotSupported(
	aten::hitable::SamplePosNormalPdfResult* result,
	Context* ctxt, 
	aten::ShapeParameter* shape,
	aten::sampler* smpl)
{
	printf("Sample PosNormalArea Not Supported[%d]\n", shape->type);
}

__device__ void getTriangleSamplePosNormalArea(
	aten::hitable::SamplePosNormalPdfResult* result,
	Context* ctxt, 
	aten::ShapeParameter* shape,
	aten::sampler* sampler)
{
	int r = sampler->nextSample();
	int basePrimIdx = aten::cmpMin(r * shape->primnum, shape->primnum - 1);

	int primidx = basePrimIdx + shape->primid;

	aten::PrimitiveParamter* prim = &ctxt->prims[primidx];

	float4 _p0 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[0] + 0);
	float4 _p1 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[1] + 0);
	float4 _p2 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[2] + 0);

	float4 _n0 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[0] + 1);
	float4 _n1 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[1] + 1);
	float4 _n2 = tex1Dfetch<float4>(ctxt->vertices, 4 * prim->idx[2] + 1);

	aten::vec3 p0 = aten::make_float3(_p0.x, _p0.y, _p0.z);
	aten::vec3 p1 = aten::make_float3(_p1.x, _p1.y, _p1.z);
	aten::vec3 p2 = aten::make_float3(_p2.x, _p2.y, _p2.z);
	
	aten::vec3 n0 = aten::make_float3(_n0.x, _n0.y, _n0.z);
	aten::vec3 n1 = aten::make_float3(_n1.x, _n1.y, _n1.z);
	aten::vec3 n2 = aten::make_float3(_n2.x, _n2.y, _n2.z);

	// 0 <= a + b <= 1
	real a = sampler->nextSample();
	real b = sampler->nextSample();

	real d = a + b;

	if (d > 1) {
		a /= d;
		b /= d;
	}

	// dSÀ•WŒn(barycentric coordinates).
	// v0Šî€.
	// p = (1 - a - b)*v0 + a*v1 + b*v2
	aten::vec3 p = (1 - a - b) * p0 + a * p1 + b * p2;

	aten::vec3 n = (1 - a - b) * n0 + a * n1 + b * n2;
	n.normalize();

	// ŽOŠpŒ`‚Ì–ÊÏ = ‚Q•Ó‚ÌŠOÏ‚Ì’·‚³ / 2;
	auto e0 = p1 - p0;
	auto e1 = p2 - p0;
	auto area = real(0.5) * cross(e0, e1).length();

	result->pos = p;
	result->nml = n;
	result->area = area;

	result->a = a;
	result->b = b;

	result->idx[0] = prim->idx[0];
	result->idx[1] = prim->idx[1];
	result->idx[2] = prim->idx[2];

	real orignalLen = (p1 - p0).length();

	auto mtxL2W = ctxt->matrices[shape->mtxid * 2 + 0];

	real scaledLen = 0;
	{
		auto v0 = mtxL2W.apply(p0);
		auto v1 = mtxL2W.apply(p1);

		scaledLen = (v1 - v0).length();
	}

	real ratio = scaledLen / orignalLen;
	ratio = ratio * ratio;

	result->area = shape->area * ratio;
}

__device__ FuncGetSamplePosNormalArea funcGetSamplePosNormalArea[aten::ShapeType::ShapeTypeMax];

__device__  void sampleAreaLight(
	aten::LightSampleResult* result,
	Context* ctxt,
	const aten::LightParameter* light,
	const aten::vec3& org,
	aten::sampler* sampler)
{
	bool isHit = false;
	aten::ShapeParameter* s = (aten::ShapeParameter*)light->object.ptr;

	aten::ray r;
	aten::hitrecord rec;
	aten::Intersection isect;

	if (sampler) {
		aten::hitable::SamplePosNormalPdfResult result;
		result.idx[0] = -1;

		const aten::ShapeParameter* realShape = (s->shapeid >= 0 ? &ctxt->shapes[s->shapeid] : s);

		funcGetSamplePosNormalArea[realShape->type](&result, ctxt, s, sampler);

		auto dir = result.pos - org;
		r = aten::ray(org, dir);

		if (result.idx[0] >= 0) {
			isect.t = dir.length();

			isect.idx[0] = result.idx[0];
			isect.idx[1] = result.idx[1];
			isect.idx[2] = result.idx[2];

			isect.a = result.a;
			isect.b = result.b;
		}
		else {
			// TODO
			// Only for sphere...
			AT_NAME::sphere::hit(s, r, AT_MATH_EPSILON, AT_MATH_INF, &isect);
		}
	}
	else {
		// TODO
		// Only for sphere...
		auto pos = s->center;
		auto dir = pos - org;
		r = aten::ray(org, dir);
		AT_NAME::sphere::hit(s, r, AT_MATH_EPSILON, AT_MATH_INF, &isect);
	}

	evalHitResult(ctxt, s, r, &rec, &isect);

	AT_NAME::AreaLight::sample(result, &rec, light, org, sampler);
}

__device__ void sampleDirectionalLight(
	aten::LightSampleResult* result,
	Context* ctxt,
	const aten::LightParameter* light,
	const aten::vec3& org,
	aten::sampler* sampler)
{
	AT_NAME::DirectionalLight::sample(result, light, org, sampler);
}

__device__ void samplePointLight(
	aten::LightSampleResult* result,
	Context* ctxt,
	const aten::LightParameter* light,
	const aten::vec3& org,
	aten::sampler* sampler)
{
	AT_NAME::PointLight::sample(result, light, org, sampler);
}

__device__ void sampleSpotLight(
	aten::LightSampleResult* result,
	Context* ctxt,
	const aten::LightParameter* light,
	const aten::vec3& org,
	aten::sampler* sampler)
{
	AT_NAME::SpotLight::sample(result, light, org, sampler);
}

__device__ void sampleLightNotSupported(
	aten::LightSampleResult* result,
	Context* ctxt,
	const aten::LightParameter* light,
	const aten::vec3& org,
	aten::sampler* sampler)
{
	printf("Sample Light Not Supported[%d]\n", light->type);
}

__device__ FuncLightSample funcSampleLight[aten::LightType::LightTypeMax];

__device__ void addLighFuncs()
{
	funcSampleLight[aten::LightType::Area] = sampleAreaLight;
	funcSampleLight[aten::LightType::IBL] = sampleLightNotSupported;
	funcSampleLight[aten::LightType::Direction] = sampleDirectionalLight;
	funcSampleLight[aten::LightType::Point] = samplePointLight;
	funcSampleLight[aten::LightType::Spot] = sampleSpotLight;

	funcGetSamplePosNormalArea[aten::ShapeType::Polygon] = getTriangleSamplePosNormalArea;
	funcGetSamplePosNormalArea[aten::ShapeType::Instance] = getSamplePosNormalAreaNotSupported;
	funcGetSamplePosNormalArea[aten::ShapeType::Sphere] = getSphereSamplePosNormalArea;;
	funcGetSamplePosNormalArea[aten::ShapeType::Cube] = getSamplePosNormalAreaNotSupported;
}

__device__ void sampleLight(
	aten::LightSampleResult* result,
	Context* ctxt,
	const aten::LightParameter* light,
	const aten::vec3& org,
	aten::sampler* sampler)
{
	funcSampleLight[light->type](result, ctxt, light, org, sampler);
}