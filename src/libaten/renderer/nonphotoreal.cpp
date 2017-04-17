#include "renderer/nonphotoreal.h"

namespace aten
{
	vec3 shadeNPR(
		material* mtrl,
		const vec3& p,
		const vec3& normal,
		real u, real v,
		scene* scene,
		sampler* sampler)
	{
		AT_ASSERT(mtrl->isNPR());
		NPRMaterial* nprMtrl = (NPRMaterial*)mtrl;

		auto light = nprMtrl->getTargetLight();
		AT_ASSERT(light);

		real cosShadow = 0;

		if (light) {
			auto sampleres = light->sample(p, sampler);

			vec3 posLight = sampleres.pos;
			vec3 nmlLight = sampleres.nml;
			real pdfLight = sampleres.pdf;

			auto lightobj = sampleres.obj;

			vec3 dirToLight = normalize(sampleres.dir);
			aten::ray shadowRay(p + dirToLight, dirToLight);

			hitrecord tmpRec;

			if (scene->hitLight(light, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, tmpRec)) {
				cosShadow = dot(normal, dirToLight);
			}
		}

		auto ret = nprMtrl->bsdf(cosShadow, u, v);

		return std::move(ret);
	}
}
