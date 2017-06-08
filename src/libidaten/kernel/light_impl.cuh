#include "kernel/idatendefs.cuh"

AT_CUDA_INLINE __device__ void getTriangleSamplePosNormalArea(
	aten::hitable::SamplePosNormalPdfResult* result,
	Context* ctxt, 
	const aten::ShapeParameter* shape,
	aten::sampler* sampler)
{
	int r = sampler->nextSample();
	int basePrimIdx = aten::cmpMin(r * shape->primnum, shape->primnum - 1);

	int primidx = basePrimIdx + shape->primid;

	aten::PrimitiveParamter* prim = &ctxt->prims[primidx];

	float4 _p0 = tex1Dfetch<float4>(ctxt->vtxPos, prim->idx[0]);
	float4 _p1 = tex1Dfetch<float4>(ctxt->vtxPos, prim->idx[1]);
	float4 _p2 = tex1Dfetch<float4>(ctxt->vtxPos, prim->idx[2]);

	float4 _n0 = tex1Dfetch<float4>(ctxt->vtxNml, prim->idx[0]);
	float4 _n1 = tex1Dfetch<float4>(ctxt->vtxNml, prim->idx[1]);
	float4 _n2 = tex1Dfetch<float4>(ctxt->vtxNml, prim->idx[2]);

	aten::vec3 p0 = aten::vec3(_p0.x, _p0.y, _p0.z);
	aten::vec3 p1 = aten::vec3(_p1.x, _p1.y, _p1.z);
	aten::vec3 p2 = aten::vec3(_p2.x, _p2.y, _p2.z);
	
	aten::vec3 n0 = aten::vec3(_n0.x, _n0.y, _n0.z);
	aten::vec3 n1 = aten::vec3(_n1.x, _n1.y, _n1.z);
	aten::vec3 n2 = aten::vec3(_n2.x, _n2.y, _n2.z);

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
	n = normalize(n);

	// ŽOŠpŒ`‚Ì–ÊÏ = ‚Q•Ó‚ÌŠOÏ‚Ì’·‚³ / 2;
	auto e0 = p1 - p0;
	auto e1 = p2 - p0;
	auto area = real(0.5) * cross(e0, e1).length();

	result->pos = p;
	result->nml = n;
	result->area = area;

	result->a = a;
	result->b = b;

	result->primid = primidx;

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

AT_CUDA_INLINE __device__  void sampleAreaLight(
	aten::LightSampleResult* result,
	Context* ctxt,
	const aten::LightParameter* light,
	const aten::vec3& org,
	aten::sampler* sampler)
{
	bool isHit = false;
	aten::ShapeParameter* s = (light->objid >= 0 ? &ctxt->shapes[light->objid] : nullptr);

	aten::ray r;
	aten::hitrecord rec;
	aten::Intersection isect;

	if (sampler) {
		aten::hitable::SamplePosNormalPdfResult result;

		const aten::ShapeParameter* realShape = (s->shapeid >= 0 ? &ctxt->shapes[s->shapeid] : s);

		if (realShape->type == aten::ShapeType::Polygon) {
			getTriangleSamplePosNormalArea(&result, ctxt, realShape, sampler);
		}
		else if (realShape->type == aten::ShapeType::Sphere) {
			AT_NAME::sphere::getSamplePosNormalArea(&result, s, sampler);
		}
		else {
			// TODO
		}

		auto dir = result.pos - org;
		r = aten::ray(org, dir);

		if (result.primid >= 0) {
			isect.t = dir.length();

			isect.primid = result.primid;

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

	result->obj = s;
}

AT_CUDA_INLINE __device__ void sampleLight(
	aten::LightSampleResult* result,
	Context* ctxt,
	const aten::LightParameter* light,
	const aten::vec3& org,
	aten::sampler* sampler)
{
	switch (light->type) {
	case aten::LightType::Area:
		sampleAreaLight(result, ctxt, light, org, sampler);
		break;
	case aten::LightType::IBL:
		// TODO
		break;
	case aten::LightType::Direction:
		AT_NAME::DirectionalLight::sample(result, light, org, sampler);
		break;
	case aten::LightType::Point:
		AT_NAME::PointLight::sample(result, light, org, sampler);
		break;
	case aten::LightType::Spot:
		AT_NAME::SpotLight::sample(result, light, org, sampler);
		break;
	}
}