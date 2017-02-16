#include "pathtracing.h"
#include "misc/thread.h"
#include "sampler/UniformDistributionSampler.h"

namespace aten
{
	// TODO
	void sampleLight(
		sampler* sampler,
		const sphere& sphere,
		vec3& posLight, 
		vec3& nmlLight,
		real& pdfLight)
	{
		auto r = sphere.radius();

		auto r1 = sampler->nextSample();
		auto r2 = sampler->nextSample();

		auto z = 1.0 - 2.0 * r2; // [0,1] -> [-1, 1]

		auto sin_theta = aten::sqrt(1 - z * z);
		auto phi = 2 * AT_MATH_PI * r1;

		auto x = aten::cos(phi) * sin_theta;
		auto y = aten::sin(phi) * sin_theta;

		vec3 dir(x, y, z);
		dir.normalize();

		auto p = dir * (r + AT_MATH_EPSILON);

		posLight = sphere.center() + p;
		nmlLight = normalize(posLight - sphere.center());

		pdfLight = 1.0 / (4.0f * AT_MATH_PI * r * r);
	}

	real sampleLightPDF(const sphere& sphere)
	{
		auto r = sphere.radius();
		real pdfLight = 1.0 / (4.0f * AT_MATH_PI * r * r);

		return pdfLight;
	}

	vec3 PathTracing::radiance(
		sampler* sampler,
		const ray& inRay,
		scene* scene)
	{
		uint32_t depth = 0;
		uint32_t maxDepth = m_maxDepth;
		uint32_t rrDepth = m_rrDepth;

		aten::ray ray = inRay;

		vec3 contribution(0, 0, 0);
		vec3 throughput(1, 1, 1);

		real pdfb;

		while (depth < maxDepth) {
			hitrecord rec;

			if (scene->hit(ray, AT_MATH_EPSILON, AT_MATH_INF, rec)) {
				// 交差位置の法線.
				// 物体からのレイの入出を考慮.
				const vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

				// Implicit conection to light.
				if (rec.mtrl->isEmissive()) {
					if (depth == 0) {
						// Ray hits the light directly.
						auto emit = rec.mtrl->color();
						return std::move(emit);
					}
					else {
						auto cosLight = dot(orienting_normal, -ray.dir);
						auto dist2 = (rec.p - ray.org).squared_length();

						if (cosLight >= 0) {
							// TODO
							auto pdfLight = sampleLightPDF(*(const sphere*)rec.obj);

							// Convert pdf area to sradian.
							// http://www.slideshare.net/h013/edubpt-v100
							// p31 - p35
							pdfLight = pdfLight * dist2 / cosLight;

							auto misW = pdfb / (pdfLight + pdfb);

							auto emit = rec.mtrl->color();

							contribution += throughput * misW * emit;

							// When ray hit the light, tracing will finish.
							break;
						}
					}
				}

				// Explicit conection to light.
				if (!rec.mtrl->isSingular())
				{
					// TODO
					auto light = scene->getLight(0);

					if (light) {
						vec3 posLight;
						vec3 nmlLight;
						real pdfLight;
						sampleLight(sampler, *light, posLight, nmlLight, pdfLight);

						vec3 dirToLight = normalize(posLight - rec.p);
						aten::ray shadowRay(rec.p, dirToLight);

						hitrecord tmpRec;

						if (scene->hit(shadowRay, AT_MATH_EPSILON, AT_MATH_INF, tmpRec)) {
							if (tmpRec.obj == light) {
								// Shadow ray hits the light.
								auto cosShadow = dot(orienting_normal, dirToLight);
								auto cosLight = dot(nmlLight, -dirToLight);
								auto dist2 = (posLight - rec.p).squared_length();

								if (cosShadow >= 0 && cosLight >= 0) {
									auto G = cosShadow * cosLight / dist2;

									auto brdf = rec.mtrl->brdf(orienting_normal, dirToLight, rec.u, rec.v);
									pdfb = rec.mtrl->pdf(orienting_normal, dirToLight);

									// Convert pdf from steradian to area.
									// http://www.slideshare.net/h013/edubpt-v100
									// p31 - p35
									pdfb = pdfb * cosLight / dist2;

									auto misW = pdfLight / (pdfb + pdfLight);

									// Get light color.
									auto emit = tmpRec.mtrl->color();

									contribution += misW * (brdf * emit * G) / pdfLight;
								}
							}
						}
					}
				}

				real russianProb = real(1);

				if (depth > rrDepth) {
					auto t = normalize(throughput);
					auto p = max(t.r, max(t.g, t.b));

					russianProb = sampler->nextSample();

					if (russianProb >= p) {
						break;
					}
					else {
						russianProb = p;
					}
				}

#if 0
				// Sample next direction.
				auto nextDir = rec.mtrl->sampleDirection(
					ray.dir,
					orienting_normal, 
					sampler);

				pdfb = rec.mtrl->pdf(orienting_normal, nextDir);

				auto brdf = rec.mtrl->brdf(orienting_normal, nextDir);

				auto c = max(dot(orienting_normal, nextDir), real(0));
#else
				auto sampling = rec.mtrl->sample(ray.dir, orienting_normal, sampler, rec.u, rec.v);

				auto nextDir = sampling.dir;
				pdfb = sampling.pdf;
				auto brdf = sampling.brdf;

				auto c = max(
					dot(sampling.into ? -orienting_normal : orienting_normal, nextDir), 
					real(0));
#endif

				throughput *= brdf * c / pdfb;
				throughput /= russianProb;

				// Make next ray.
				ray = aten::ray(rec.p, nextDir);
			}
			else {
				// TODO
				// Background.
				auto bg = sampleBG(ray);
				return contribution + throughput * bg;
			}

			depth++;
		}

		return contribution;
	}

	void PathTracing::render(
		Destination& dst,
		scene* scene,
		camera* camera)
	{
		int width = dst.width;
		int height = dst.height;
		uint32_t sample = dst.sample;
		vec3* color = dst.buffer;

		m_maxDepth = dst.maxDepth;
		m_rrDepth = dst.russianRouletteDepth;

		if (m_rrDepth > m_maxDepth) {
			m_rrDepth = m_maxDepth - 1;
		}

#ifdef ENABLE_OMP
#pragma omp parallel
#endif
		{
			auto idx = thread::getThreadIdx();

			XorShift rnd(idx);
			UniformDistributionSampler sampler(&rnd);

#ifdef ENABLE_OMP
#pragma omp for
//#pragma omp parallel for
#endif
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int pos = y * width + x;

					vec3 col;

					for (uint32_t i = 0; i < sample; i++) {
						real u = real(x + sampler.nextSample()) / real(width);
						real v = real(y + sampler.nextSample()) / real(height);

						auto ray = camera->sample(u, v);

						col += radiance(&sampler, ray, scene);
					}

					col /= (real)sample;

					color[pos] = col;
				}
			}
		}
	}
}
