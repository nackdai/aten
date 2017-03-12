#include "renderer/geominforender.h"
#include "sampler/xorshift.h"
#include "sampler/UniformDistributionSampler.h"

namespace aten
{
	GeometryInfoRenderer::Path GeometryInfoRenderer::radiance(
		const ray& inRay,
		scene* scene)
	{
		hitrecord rec;
		ray ray = inRay;

		Path path;

		if (scene->hit(ray, AT_MATH_EPSILON, AT_MATH_INF, rec)) {
			// 交差位置の法線.
			// 物体からのレイの入出を考慮.
			vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

			// Apply normal map.
			rec.mtrl->applyNormalMap(orienting_normal, orienting_normal, rec.u, rec.v);
			path.normal = orienting_normal;

			if (rec.mtrl->isEmissive()) {
				// Ray hits the light directly.
				path.albedo = rec.mtrl->color();
			}
			else {
				path.albedo = rec.mtrl->color();
				path.albedo *= rec.mtrl->sampleAlbedoMap(rec.u, rec.v);
			}
		}
		else {
			auto ibl = scene->getIBL();
			if (ibl) {
				auto bg = ibl->getEnvMap()->sample(ray);
				path.albedo = bg;
			}
			else {
				auto bg = sampleBG(ray);
				path.albedo = bg;
			}
		}

		return std::move(path);
	}

	void GeometryInfoRenderer::render(
		Destination& dst,
		scene* scene,
		camera* camera)
	{
		int width = dst.width;
		int height = dst.height;

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int pos = y * width + x;

				// Dummy. Not use...
				XorShift rnd(0);
				UniformDistributionSampler sampler(&rnd);

				real u = real(x + 0.5) / real(width);
				real v = real(y + 0.5) / real(height);

				auto camsample = camera->sample(u, v, &sampler);

				auto path = radiance(camsample.r, scene);

				if (dst.geominfo.normal) {
					dst.geominfo.normal[pos] = path.normal;
				}
				if (dst.geominfo.depth) {
					dst.geominfo.depth[pos] = vec3(path.depth);
				}
				if (dst.geominfo.albedo) {
					dst.geominfo.albedo[pos] = path.albedo;
				}
			}
		}
	}
}
