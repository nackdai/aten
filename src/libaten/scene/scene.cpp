#include "scene/scene.h"

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

		if (light->isSingular()) {
			if (!isHit) {
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

		static const vec3 RGB2Y(0.29900, 0.58700, 0.11400);

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

			auto cosShadow = aten::abs(dot(nml, dirToLight));
			auto dist2 = lightsample.dir.squared_length();
			auto dist = aten::sqrt(dist2);

			auto y = dot(RGB2Y, lightsample.finalColor);

			if (light->isSingular()) {
				costs[i] = y * cosShadow / pdfLight;
			}
			else {
				costs[i] = y * cosShadow / dist2 / pdfLight;
			}
			sumCost += costs[i];
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

		AT_ASSERT(false);
		return nullptr;
#endif
	}
}
