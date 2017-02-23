#include "pathtracing.h"
#include "misc/thread.h"
#include "sampler/xorshift.h"
#include "sampler/halton.h"
#include "sampler/sobolproxy.h"
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

		posLight = sphere.getRandomPosOn(sampler);
		nmlLight = normalize(posLight - sphere.center());

		pdfLight = 1.0 / (4.0f * AT_MATH_PI * r * r);
	}

	real sampleLightPDF(const sphere& sphere)
	{
		auto r = sphere.radius();
		real pdfLight = 1.0 / (4.0f * AT_MATH_PI * r * r);

		return pdfLight;
	}

	// NOTE
	// https://www.slideshare.net/shocker_0x15/ss-52688052

	vec3 PathTracing::radiance(
		sampler* sampler,
		const ray& inRay,
		camera* cam,
		CameraSampleResult& camsample,
		scene* scene)
	{
		uint32_t depth = 0;
		uint32_t maxDepth = m_maxDepth;
		uint32_t rrDepth = m_rrDepth;

		aten::ray ray = inRay;

		vec3 contribution(0, 0, 0);
		vec3 throughput(1, 1, 1);

		auto Wdash = cam->getWdash(
			camsample.posOnImageSensor,
			camsample.posOnLens,
			camsample.posOnObjectplane);

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
							//auto pdfLight = sampleLightPDF(*(const sphere*)rec.obj);
							auto pdfLight = 1 / rec.area;

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

				if (depth == 0) {
					auto areaPdf = cam->getPdfImageSensorArea(rec.p, orienting_normal);
					
					//throughput *= Wdash;
					throughput /= areaPdf;
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

									auto brdf = rec.mtrl->brdf(orienting_normal, ray.dir, dirToLight, rec.u, rec.v);
									pdfb = rec.mtrl->pdf(orienting_normal, ray.dir, dirToLight);

									if (pdfb > real(0)) {
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
				}

				real russianProb = real(1);

				if (depth > rrDepth) {
					auto t = normalize(throughput);
					auto p = std::max(t.r, std::max(t.g, t.b));

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

				auto c = std::max(dot(orienting_normal, nextDir), real(0));
#else
				auto sampling = rec.mtrl->sample(ray.dir, orienting_normal, rec, sampler, rec.u, rec.v);

				auto nextDir = sampling.dir;
				pdfb = sampling.pdf;
				auto brdf = sampling.brdf;

				// TODO
				// AMDのはabsしているが、正しい?
				//auto c = dot(orienting_normal, nextDir);
				auto c = aten::abs(dot(orienting_normal, nextDir));
#endif

				if (pdfb > 0) {
					throughput *= brdf * c / pdfb;
					throughput /= russianProb;
				}
				else {
					break;
				}

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

	inline bool isInvalidColor(const vec3& v)
	{
		bool b = isInvalid(v);
		if (!b) {
			if (v.x < 0 || v.y < 0 || v.z < 0) {
				b = true;
			}
		}

		return b;
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

		const vec3 camera_position = vec3(50.0, 52.0, 220.0);
		const vec3 camera_dir = normalize(vec3(0.0, -0.04, -1.0));
		const vec3 camera_up = vec3(0.0, 1.0, 0.0);

		// ワールド座標系でのスクリーンの大きさ
		const double screen_width = 30.0 * width / height;
		const double screen_height = 30.0;
		// スクリーンまでの距離
		const double screen_dist = 40.0;
		// スクリーンを張るベクトル
		const vec3 screen_x = normalize(cross(camera_dir, camera_up)) * screen_width;
		const vec3 screen_y = normalize(cross(screen_x, camera_dir)) * screen_height;
		const vec3 screen_center = camera_position + camera_dir * screen_dist;

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

			//XorShift rnd(idx);
			//UniformDistributionSampler sampler(&rnd);

#ifdef ENABLE_OMP
#pragma omp for
//#pragma omp parallel for
#endif
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int pos = y * width + x;

					vec3 col;

					for (uint32_t i = 0; i < sample; i++) {
						XorShift rnd((y * height * 4 + x * 4) * sample + i + 1);
						//Halton rnd((y * height * 4 + x * 4) * sample + i + 1);
						//Sobol rnd((y * height * 4 + x * 4) * sample + i + 1);
						UniformDistributionSampler sampler(&rnd);

						real u = real(x + sampler.nextSample()) / real(width);
						real v = real(y + sampler.nextSample()) / real(height);

						auto camsample = camera->sample(u, v, &sampler);

						auto ray = camsample.r;

						auto L = radiance(
							&sampler, 
							ray, 
							camera,
							camsample,
							scene);

						if (isInvalidColor(L)) {
							AT_PRINTF("Invalid(%d/%d)\n", x, y);
						}

						auto pdfOnImageSensor = camsample.pdfOnImageSensor;
						auto pdfOnLens = camsample.pdfOnLens;

						auto s = camera->getSensitivity(
							camsample.posOnImageSensor,
							camsample.posOnLens);

						col += L * s / (pdfOnImageSensor * pdfOnLens);
					}

					col /= (real)sample;

					color[pos] = col;
				}
			}
		}
	}
}
