#include "pathtracing.h"
#include "misc/thread.h"
#include "sampler/UniformDistributionSampler.h"

namespace aten
{
	vec3 radiance(
		sampler& sampler,
		const ray& inRay,
		scene* scene,
		uint32_t maxDepth)
	{
		uint32_t depth = 0;

		aten::ray ray = inRay;

		vec3 throughput(1, 1, 1);

		while (depth < maxDepth) {
			hitrecord rec;

			if (scene->hit(ray, AT_MATH_EPSILON, AT_MATH_INF, rec)) {
				if (rec.mtrl->isEmissive()) {
					auto emit = rec.mtrl->color();

					vec3 result = throughput * emit;

					return std::move(result);
				}

				// 交差位置の法線.
				// 物体からのレイの入出を考慮.
				const vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

				// Sample next direction.
				auto nextDir = rec.mtrl->sampleDirection(orienting_normal, sampler);

				auto pdf = rec.mtrl->pdf(orienting_normal, nextDir);

				auto brdf = rec.mtrl->brdf(orienting_normal, nextDir);

				auto c = max(dot(orienting_normal, nextDir), CONST_REAL(0.0));

				throughput *= brdf * c / pdf;

				ray = aten::ray(rec.p, nextDir);
			}
			else {
				break;
			}

			depth++;
		}

		// TODO
		return vec3();
	}

	void PathTracing::render(
		Destination& dst,
		scene* scene,
		camera* camera)
	{
		int width = dst.width;
		int height = dst.height;
		uint32_t maxDepth = dst.maxDepth;
		uint32_t sample = dst.sample;
		vec3* color = dst.buffer;

#ifdef ENABLE_OMP
#pragma omp parallel
#endif
		{
			auto idx = thread::getThreadIdx();

			XorShift rnd(idx);
			UniformDistributionSampler sampler(rnd);

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

						col += radiance(sampler, ray, scene, maxDepth);
					}

					col /= (real)sample;

					color[pos] = col;
				}
			}
		}
	}
}
