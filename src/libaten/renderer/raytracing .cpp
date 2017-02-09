#include "raytracing.h"
#include "sampler/UniformDistributionSampler.h"

namespace aten
{
	vec3 radiance(
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

		// TODO
		return vec3();
	}

	void RayTracing::render(
		Destination& dst,
		scene* scene,
		camera* camera)
	{
		int width = dst.width;
		int height = dst.height;
		uint32_t sample = dst.sample;
		vec3* color = dst.buffer;

		XorShift rnd;
		UniformDistributionSampler sampler(rnd);

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int pos = y * width + x;

				vec3 col;

				for (uint32_t i = 0; i < sample; i++) {
					real u = real(x + sampler.nextSample()) / real(width);
					real v = real(y + sampler.nextSample()) / real(height);

					auto ray = camera->sample(u, v);

					col += radiance(ray, scene);
				}

				col /= (real)sample;

				color[pos] = col;
			}
		}
	}
}
