#include "kernel/light.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"

#include "aten4idaten.h"

typedef void(*FuncLightSample)(
	aten::LightSampleResult*, const aten::LightParameter*, const aten::vec3&, aten::sampler*);

__device__  void sampleAreaLight(
	aten::LightSampleResult* result,
	const aten::LightParameter* light,
	const aten::vec3& org,
	aten::sampler* sampler)
{
	auto funcHitTestSphere = [] AT_DEVICE_API(const aten::vec3& o, const aten::UnionIdxPtr& object, aten::vec3& pos, aten::sampler* smpl, aten::hitrecord* _rec)
	{
		aten::ShapeParameter* s = (aten::ShapeParameter*)object.ptr;

		pos = s->center;

		auto dir = pos - o;
		auto dist = dir.length();

		aten::ray r(o, dir);
		bool isHit = AT_NAME::sphere::hit(s, r, AT_MATH_EPSILON, AT_MATH_INF, _rec);

		return isHit;
	};
	AT_NAME::AreaLight::sample(funcHitTestSphere, result, light, org, sampler);
}

__device__ void sampleLightNotSupported(
	aten::LightSampleResult* result,
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
	funcSampleLight[aten::LightType::Direction] = AT_NAME::DirectionalLight::sample;
	funcSampleLight[aten::LightType::Point] = AT_NAME::PointLight::sample;
	funcSampleLight[aten::LightType::Spot] = AT_NAME::SpotLight::sample;
}

__device__ void sampleLight(
	aten::LightSampleResult* result,
	const aten::LightParameter* light,
	const aten::vec3& org,
	aten::sampler* sampler)
{
	funcSampleLight[light->type](result, light, org, sampler);
}