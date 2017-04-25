#include "kernel/light.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda/helper_math.h"

#include "aten4idaten.h"

typedef aten::LightSampleResult(*FuncLightSample)(const aten::LightParameter&, const aten::vec3&, aten::sampler*);

__device__ aten::LightSampleResult sampleAreaLight(
	const aten::LightParameter& light,
	const aten::vec3& org,
	aten::sampler* sampler)
{
	auto funcHitTestSphere = [] AT_DEVICE_API(const aten::vec3& o, const aten::UnionIdxPtr& object, aten::vec3& pos, aten::sampler* smpl, aten::hitrecord& _rec)
	{
		aten::ShapeParameter* s = (aten::ShapeParameter*)object.ptr;

		pos = s->center;

		auto dir = pos - o;
		auto dist = dir.length();

		aten::ray r(o, normalize(dir));
		bool isHit = AT_NAME::sphere::hit(*s, r, AT_MATH_EPSILON, AT_MATH_INF, _rec);

		return isHit;
	};
	auto ret = AT_NAME::AreaLight::sample(funcHitTestSphere, light, org, sampler);
	return std::move(ret);
}

__device__ FuncLightSample funcSampleLight[aten::LightType::LightTypeMax];

__device__ void addLighFuncs()
{
	funcSampleLight[aten::LightType::Area] = sampleAreaLight;
	funcSampleLight[aten::LightType::IBL] = nullptr;
	funcSampleLight[aten::LightType::Direction] = AT_NAME::DirectionalLight::sample;
	funcSampleLight[aten::LightType::Point] = AT_NAME::PointLight::sample;
	funcSampleLight[aten::LightType::Spot] = AT_NAME::SpotLight::sample;
}

__device__ aten::LightSampleResult sampleLight(
	const aten::LightParameter& light,
	const aten::vec3& org,
	aten::sampler* sampler)
{
	aten::LightSampleResult ret;

	if (light.type == aten::LightType::IBL) {
		// TODO
	}
	else {
		ret = funcSampleLight[light.type](light, org, sampler);
	}
	return std::move(ret);
}