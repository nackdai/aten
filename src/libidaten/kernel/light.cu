#include "kernel/light.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten.h"

typedef aten::LightSampleResult(*FuncLightSample)(const aten::LightParameter&, const aten::vec3&, aten::sampler*);

__device__ aten::LightSampleResult sampleLight(
	const aten::LightParameter& light,
	const aten::vec3& org,
	aten::sampler* sampler)
{
	constexpr FuncLightSample funcs[] = {
		nullptr,
		nullptr,
		aten::DirectionalLight::sample,
		aten::PointLight::sample,
		aten::SpotLight::sample,
	};

	aten::LightSampleResult ret;

	if (light.type == aten::LightType::IBL) {
		// TODO
	}
	else if (light.type == aten::LightType::Area) {
		auto funcHitTestSphere = [] AT_DEVICE_API(const aten::vec3& o, const aten::UnionIdxPtr& object, aten::vec3& pos, aten::sampler* smpl, aten::hitrecord& _rec)
		{
			aten::ShapeParameter* s = (aten::ShapeParameter*)object.ptr;

			pos = s->center;

			auto dir = pos - o;
			auto dist = dir.length();

			aten::ray r(o, normalize(dir));
			bool isHit = aten::sphere::hit(*s, r, AT_MATH_EPSILON, AT_MATH_INF, _rec);

			return isHit;
		};
		ret = aten::AreaLight::sample(funcHitTestSphere, light, org, sampler);
	}
	else {
		ret = funcs[light.type](light, org, sampler);
	}
	return std::move(ret);
}