#include "raytracing.h"
#include "misc/thread.h"
#include "renderer/nonphotoreal.h"

namespace aten
{
	vec3 RayTracing::radiance(
		const ray& inRay,
		scene* scene)
	{
		uint32_t depth = 0;

		aten::ray ray = inRay;

		vec3 contribution(0, 0, 0);
		vec3 throughput(1, 1, 1);

		while (depth < m_maxDepth) {
			hitrecord rec;

			if (scene->hit(ray, AT_MATH_EPSILON, AT_MATH_INF, rec)) {
				if (rec.mtrl->isEmissive()) {
					auto emit = rec.mtrl->color();
					contribution += throughput * emit;
					return std::move(contribution);
				}

				// 交差位置の法線.
				// 物体からのレイの入出を考慮.
				const vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

				if (rec.mtrl->isSingular() || rec.mtrl->isTranslucent()) {
					auto sampling = rec.mtrl->sample(ray, orienting_normal, rec, nullptr, rec.u, rec.v);

					auto nextDir = normalize(sampling.dir);
					auto bsdf = sampling.bsdf;

					throughput *= bsdf;

					// Make next ray.
					ray = aten::ray(rec.p, nextDir);
				}
				else if (rec.mtrl->isNPR()) {
					// Non-Photo-Real.
					contribution = shadeNPR(rec.mtrl, rec.p, orienting_normal, rec.u, rec.v, scene, nullptr);
					return std::move(contribution);
				}
				else {
					auto lightNum = scene->lightNum();

					for (int i = 0; i < lightNum; i++) {
						auto light = scene->getLight(i);

						if (light->isIBL()) {
							continue;
						}

						auto sampleres = light->sample(rec.p, nullptr);

						vec3 dirToLight = sampleres.dir;
						auto len = dirToLight.length();

						dirToLight.normalize();

						auto lightobj = sampleres.obj;

						auto albedo = rec.mtrl->color();

						aten::ray shadowRay(rec.p, dirToLight);

						hitrecord tmpRec;

						if (scene->hitLight(light, sampleres.pos, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, tmpRec)) {
							auto lightColor = sampleres.finalColor;

							if (light->isInfinite()) {
								len = real(1);
							}

							const auto c0 = std::max(real(0.0), dot(orienting_normal, dirToLight));
							real c1 = real(1);

							if (!light->isSingular()) {
								c1 = std::max(real(0.0), dot(sampleres.nml, -dirToLight));
							}

							auto G = c0 * c1 / (len * len);

							contribution += throughput * (albedo * lightColor) * G;
						}
					}

					break;
				}
			}
			else {
				auto ibl = scene->getIBL();

				if (ibl) {
					auto bg = ibl->getEnvMap()->sample(ray);
					contribution += throughput * bg;
				}
				else {
					auto bg = sampleBG(ray);
					contribution += throughput * bg;
				}

				return std::move(contribution);
			}

			depth++;
		}

		return contribution;
	}

	void RayTracing::render(
		Destination& dst,
		scene* scene,
		camera* camera)
	{
		int width = dst.width;
		int height = dst.height;

		m_maxDepth = dst.maxDepth;

		uint32_t sample = 1;

#ifdef ENABLE_OMP
#pragma omp parallel
#endif
		{
#ifdef ENABLE_OMP
#pragma omp for
#endif
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int pos = y * width + x;

					real u = (real(x) + real(0.5)) / real(width - 1);
					real v = (real(y) + real(0.5)) / real(height - 1);

					auto camsample = camera->sample(u, v, nullptr);

					auto col = radiance(camsample.r, scene);

					dst.buffer->put(x, y, vec4(col, 1));
				}
			}
		}
	}
}
