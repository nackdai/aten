#include "raytracing.h"
#include "misc/thread.h"
#include "sampler/UniformDistributionSampler.h"

namespace aten
{
	vec3 RayTracing::radiance(
		const ray& ray,
		scene* scene)
	{
		hitrecord rec;

		if (scene->hit(ray, AT_MATH_EPSILON, AT_MATH_INF, rec)) {
			if (rec.mtrl->isEmissive()) {
				auto emit = rec.mtrl->color();
				return std::move(emit);
			}

			vec3 color = rec.mtrl->color();

			return std::move(color);
		}
		
		auto bg = sampleBG(ray);

		return std::move(bg);
	}

	void RayTracing::render(
		Destination& dst,
		scene* scene,
		camera* camera)
	{
		int width = dst.width;
		int height = dst.height;
		vec3* color = dst.buffer;

		uint32_t sample = 1;

#ifdef ENABLE_OMP
#pragma omp parallel
#endif
		{
			auto idx = thread::getThreadIdx();

			XorShift rnd(idx);
			UniformDistributionSampler sampler(rnd);

#ifdef ENABLE_OMP
#pragma omp for
#endif
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int pos = y * width + x;

					real u = real(x) / real(width - 1);
					real v = real(y) / real(height - 1);

					auto ray = camera->sample(u, v);

					auto col = radiance(ray, scene);

					color[pos] = col;
				}
			}
		}
	}
}
