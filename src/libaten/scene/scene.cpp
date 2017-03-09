#include "scene/scene.h"
#include "misc/color.h"

namespace aten {
	bool scene::hitLight(
		const Light* light,
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec)
	{
		bool isHit = hit(r, t_min, t_max, rec);

		auto lightobj = light->getLightObject();

		if (lightobj) {
			if (isHit && rec.obj == lightobj) {
				return true;
			}
		}

		if (light->isInifinite()) {
			if (isHit) {
				// Hit something.
				return false;
			}
			else {
				// Hit nothing.
				return true;
			}
		}
		else if (light->isSingular()) {
			const auto& lightpos = light->getPos();
			auto distToLight = (lightpos - r.org).length();

			if (isHit && rec.t < distToLight) {
				// Ray hits something, and the distance to the object is near than the distance to the light.
				return false;
			}
			else {
				// Ray don't hit anything, or the distance to the object is far than the distance to the light.
				return true;
			}
		}

		return false;
	}

	Light* scene::sampleLight(
		const vec3& org,
		const vec3& nml,
		sampler* sampler,
		real& selectPdf,
		LightSampleResult& sampleRes)
	{
#if 0
		Light* light = nullptr;

		auto num = m_lights.size();
		if (num > 0) {
			auto r = sampler->nextSample();
			uint32_t idx = (uint32_t)aten::clamp<real>(r * num, 0, num - 1);
			light = m_lights[idx];

			sampleRes = light->sample(org, sampler);
			selectPdf = real(1) / num;
		}
		else {
			selectPdf = 1;
		}

		return light;
#else
		// Resampled Importance Sampling.
		// For reducing variance...maybe...

		std::vector<LightSampleResult> samples(m_lights.size());
		std::vector<real> costs(m_lights.size());

		real sumCost = 0;

		for (int i = 0; i < m_lights.size(); i++) {
			const auto light = m_lights[i];

			samples[i] = light->sample(org, sampler);

			const auto& lightsample = samples[i];

			vec3 posLight = lightsample.pos;
			vec3 nmlLight = lightsample.nml;
			real pdfLight = lightsample.pdf;
			vec3 dirToLight = normalize(lightsample.dir);

			auto cosShadow = dot(nml, dirToLight);
			auto dist2 = lightsample.dir.squared_length();
			auto dist = aten::sqrt(dist2);

			auto illum = color::luminance(lightsample.finalColor);

			if (cosShadow > 0) {
				if (light->isSingular() || light->isInifinite()) {
					costs[i] = illum * cosShadow / pdfLight;
				}
				else {
					costs[i] = illum * cosShadow / dist2 / pdfLight;
				}
				sumCost += costs[i];
			}
			else {
				costs[i] = 0;
			}
		}

		auto r = sampler->nextSample() * sumCost;
		
		real sum = 0;

		for (int i = 0; i < costs.size(); i++) {
			const auto c = costs[i];
			sum += c;

			if (r <= sum && c > 0) {
				auto light = m_lights[i];
				sampleRes = samples[i];
				selectPdf = c / sumCost;
				return light;
			}
		}

		return nullptr;
#endif
	}
}
