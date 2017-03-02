#include "pathtracing.h"
#include "misc/thread.h"
#include "sampler/xorshift.h"
#include "sampler/halton.h"
#include "sampler/sobolproxy.h"
#include "sampler/UniformDistributionSampler.h"

#include "primitive/sphere.h"

namespace aten
{
	// NOTE
	// https://www.slideshare.net/shocker_0x15/ss-52688052

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

	PathTracing::Path PathTracing::radiance(
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

		vec3 throughput(1, 1, 1);

		auto Wdash = cam->getWdash(
			camsample.posOnImageSensor,
			camsample.posOnLens,
			camsample.posOnObjectplane);

		real pdfb = 0;
		material* prevMtrl = nullptr;

		Path path;

		while (depth < maxDepth) {
			hitrecord rec;

			if (scene->hit(ray, AT_MATH_EPSILON, AT_MATH_INF, rec)) {
				// 交差位置の法線.
				// 物体からのレイの入出を考慮.
				vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

				// Implicit conection to light.
				if (rec.mtrl->isEmissive()) {
					if (depth == 0) {
						// Ray hits the light directly.
						path.contrib = rec.mtrl->color();
						path.isTerminate = true;
						break;
					}
					else if (prevMtrl->isSingular()) {
						auto emit = rec.mtrl->color();
						path.contrib += throughput * emit;
						break;
					}
					else {
						auto cosLight = dot(orienting_normal, -ray.dir);
						auto dist2 = (rec.p - ray.org).squared_length();

						if (cosLight >= 0) {
							auto pdfLight = 1 / rec.area;

							// Convert pdf area to sradian.
							// http://www.slideshare.net/h013/edubpt-v100
							// p31 - p35
							pdfLight = pdfLight * dist2 / cosLight;

							auto misW = pdfb / (pdfLight + pdfb);

							auto emit = rec.mtrl->color();

							path.contrib += throughput * misW * emit;

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

				// Apply normal map.
				rec.mtrl->applyNormalMap(orienting_normal, orienting_normal, rec.u, rec.v);

				// Explicit conection to light.
				if (!rec.mtrl->isSingular())
				{
					real lightSelectPdf = 1;
					LightSampleResult sampleres;

					auto light = scene->sampleLight(
						rec.p,
						orienting_normal,
						sampler,
						lightSelectPdf, sampleres);

					if (light) {
						vec3 posLight = sampleres.pos;
						vec3 nmlLight = sampleres.nml;
						real pdfLight = sampleres.pdf;

						auto lightobj = sampleres.obj;

						vec3 dirToLight = normalize(sampleres.dir);
						aten::ray shadowRay(rec.p + AT_MATH_EPSILON * dirToLight, dirToLight);

						hitrecord tmpRec;

						if (scene->hitLight(light, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, tmpRec)) {
							// Shadow ray hits the light.
							auto cosShadow = dot(orienting_normal, dirToLight);
							auto dist2 = sampleres.dir.squared_length();
							auto dist = aten::sqrt(dist2);

							auto bsdf = rec.mtrl->bsdf(orienting_normal, ray.dir, dirToLight, rec.u, rec.v);
							pdfb = rec.mtrl->pdf(orienting_normal, ray.dir, dirToLight, rec.u, rec.v);

							// Get light color.
							auto emit = sampleres.finalColor;

							if (light->isSingular() || light->isInifinite()) {
								if (pdfLight > real(0)) {
									// TODO
									// ジオメトリタームの扱いについて.
									// singular light の場合は、finalColor に距離の除算が含まれている.
									// inifinite light の場合は、無限遠方になり、pdfLightに含まれる距離成分と打ち消しあう？.
									// （打ち消しあうので、pdfLightには距離成分は含んでいない）.
									auto misW = pdfLight / (pdfb + pdfLight);
									path.contrib += misW * bsdf * emit * cosShadow / pdfLight;
									path.contrib /= lightSelectPdf;
								}
							}
							else {
								auto cosLight = dot(nmlLight, -dirToLight);

								if (cosShadow >= 0 && cosLight >= 0) {
									auto G = cosShadow * cosLight / dist2;

									if (pdfb > real(0) && pdfLight > real(0)) {
										// Convert pdf from steradian to area.
										// http://www.slideshare.net/h013/edubpt-v100
										// p31 - p35
										pdfb = pdfb * cosLight / dist2;

										auto misW = pdfLight / (pdfb + pdfLight);

										path.contrib += misW * (bsdf * emit * G) / pdfLight;
										path.contrib /= lightSelectPdf;
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

				auto sampling = rec.mtrl->sample(ray.dir, orienting_normal, rec, sampler, rec.u, rec.v);

				auto nextDir = normalize(sampling.dir);
				pdfb = sampling.pdf;
				auto bsdf = sampling.bsdf;

#if 1
				real c = 1;
				if (!rec.mtrl->isSingular()) {
					// TODO
					// AMDのはabsしているが....
					//c = aten::abs(dot(orienting_normal, nextDir));
					c = dot(orienting_normal, nextDir);
				}
#else
				auto c = dot(orienting_normal, nextDir);
#endif

				//if (pdfb > 0) {
				if (pdfb > 0 && c > 0) {
					throughput *= bsdf * c / pdfb;
					throughput /= russianProb;
				}
				else {
					break;
				}

				prevMtrl = rec.mtrl;

				// Make next ray.
				ray = aten::ray(rec.p + AT_MATH_EPSILON * nextDir, nextDir);
			}
			else {
				auto ibl = scene->getIBL();
				if (ibl) {
					if (depth == 0) {
						auto bg = ibl->getEnvMap()->sample(ray);
						path.contrib += throughput * bg;
						path.isTerminate = true;
					}
					else {
						auto pdfLight = ibl->samplePdf(ray);
						auto misW = pdfb / (pdfLight + pdfb);
						auto emit = ibl->getEnvMap()->sample(ray);
						path.contrib += throughput * misW * emit;
					}
				}
				else {
					auto bg = sampleBG(ray);
					path.contrib += throughput * bg;
				}

				break;
			}

			depth++;
		}

		return std::move(path);
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

			//XorShift rnd(idx);
			//UniformDistributionSampler sampler(&rnd);

#ifdef ENABLE_OMP
#pragma omp for
#endif
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int pos = y * width + x;

					if (y == 146 && x == 311) {
						int xxx = 0;
					}

					vec3 col;
					uint32_t cnt = 0;

					for (uint32_t i = 0; i < sample; i++) {
						//XorShift rnd((y * height * 4 + x * 4) * sample + i + 1);
						//Halton rnd((y * height * 4 + x * 4) * sample + i + 1);
						Sobol rnd((y * height * 4 + x * 4) * sample + i + 1);
						UniformDistributionSampler sampler(&rnd);

						real u = real(x + sampler.nextSample()) / real(width);
						real v = real(y + sampler.nextSample()) / real(height);

						auto camsample = camera->sample(u, v, &sampler);

						auto ray = camsample.r;

						auto path = radiance(
							&sampler, 
							ray, 
							camera,
							camsample,
							scene);

						if (isInvalidColor(path.contrib)) {
							AT_PRINTF("Invalid(%d/%d[%d])\n", x, y, i);
							continue;
						}

						auto pdfOnImageSensor = camsample.pdfOnImageSensor;
						auto pdfOnLens = camsample.pdfOnLens;

						auto s = camera->getSensitivity(
							camsample.posOnImageSensor,
							camsample.posOnLens);

						col += path.contrib * s / (pdfOnImageSensor * pdfOnLens);
						cnt++;

						if (path.isTerminate) {
							break;
						}
					}

					col /= (real)cnt;

					color[pos] = col;
				}
			}
		}
	}
}
