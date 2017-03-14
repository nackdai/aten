#include "raytracing.h"
#include "misc/thread.h"
#include "sampler/xorshift.h"
#include "sampler/UniformDistributionSampler.h"

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
					auto sampling = rec.mtrl->sample(ray.dir, orienting_normal, rec, nullptr, rec.u, rec.v);

					auto nextDir = normalize(sampling.dir);
					auto bsdf = sampling.bsdf;

					throughput *= bsdf;

					// Make next ray.
					ray = aten::ray(rec.p + AT_MATH_EPSILON * nextDir, nextDir);
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

						if (scene->hit(shadowRay, AT_MATH_EPSILON, AT_MATH_INF, tmpRec)) {
							if (tmpRec.obj == lightobj) {
								const auto lightColor = sampleres.le;
								contribution += throughput * std::max(0.0, dot(orienting_normal, dirToLight)) * (albedo * lightColor) / (len * len);
							}
						}
					}
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
		vec3* color = dst.buffer;

		m_maxDepth = dst.maxDepth;

		uint32_t sample = 1;

#ifdef ENABLE_OMP
#pragma omp parallel
#endif
		{
			auto idx = thread::getThreadIdx();

			XorShift rnd(idx);
			UniformDistributionSampler sampler(&rnd);

#ifdef ENABLE_OMP
#pragma omp for
#endif
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int pos = y * width + x;

					real u = (real(x) + 0.5) / real(width - 1);
					real v = (real(y) + 0.5) / real(height - 1);

					auto camsample = camera->sample(u, v, nullptr);

					auto col = radiance(camsample.r, scene);

					color[pos] = col;
				}
			}
		}
	}
}
