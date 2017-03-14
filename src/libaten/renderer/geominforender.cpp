#include "renderer/geominforender.h"
#include "sampler/xorshift.h"
#include "sampler/UniformDistributionSampler.h"

namespace aten
{
	GeometryInfoRenderer::Path GeometryInfoRenderer::radiance(
		const ray& inRay,
		scene* scene)
	{
		ray ray = inRay;

		const auto maxDepth = m_maxDepth;
		uint32_t depth = 0;

		vec3 throughput = vec3(1);

		Path path;

		while (depth < maxDepth) {
			hitrecord rec;

			if (scene->hit(ray, AT_MATH_EPSILON, AT_MATH_INF, rec)) {
				// 交差位置の法線.
				// 物体からのレイの入出を考慮.
				vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

				// Apply normal map.
				rec.mtrl->applyNormalMap(orienting_normal, orienting_normal, rec.u, rec.v);

				if (depth == 0) {
					path.normal = orienting_normal;
					path.depth = rec.t;
				}

				if (rec.mtrl->isEmissive()) {
					path.albedo = rec.mtrl->color();
					path.albedo *= throughput;
					break;
				}
				else if (rec.mtrl->isSingular()) {
					auto sample = rec.mtrl->sample(ray.dir, orienting_normal, rec, nullptr, rec.u, rec.v);

					const auto& nextDir = sample.dir;
					throughput *= sample.bsdf;

					// Make next ray.
					ray = aten::ray(rec.p + AT_MATH_EPSILON * nextDir, nextDir);
				}
				else {
					path.albedo = rec.mtrl->color();
					path.albedo *= rec.mtrl->sampleAlbedoMap(rec.u, rec.v);
					path.albedo *= throughput;
					break;
				}
			}
			else {
				auto ibl = scene->getIBL();
				if (ibl) {
					auto bg = ibl->getEnvMap()->sample(ray);
					path.albedo = throughput * bg;
				}
				else {
					auto bg = sampleBG(ray);
					path.albedo = throughput * bg;
				}

				if (depth == 0) {
					// Far far away...
					path.depth = AT_MATH_INF;
				}

				break;
			}

			depth++;
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

		m_maxDepth = dst.maxDepth;

		real depthNorm = 1 / dst.geominfo.depthMax;

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
					if (dst.geominfo.needNormalize) {
						// [-1, 1] -> [0, 1]
						auto normal = (path.normal + 1) * 0.5;

						dst.geominfo.normal[pos] = normal;
					}
					else {
						dst.geominfo.normal[pos] = path.normal;
					}
				}
				if (dst.geominfo.depth) {
					// [-∞, ∞] -> [-d, d]
					real depth = std::min(aten::abs(path.depth), dst.geominfo.depthMax);
					depth *= path.depth < 0 ? -1 : 1;

					if (dst.geominfo.needNormalize) {
						// [-d, d] -> [-1, 1]
						depth *= depthNorm;

						// [-1, 1] -> [0, 1]
						depth = (depth + 1) * 0.5;
					}
					
					dst.geominfo.depth[pos] = vec3(depth);
				}
				if (dst.geominfo.albedo) {
					if (dst.geominfo.needNormalize) {
						path.albedo.x = std::min<real>(path.albedo.x, 1);
						path.albedo.y = std::min<real>(path.albedo.y, 1);
						path.albedo.z = std::min<real>(path.albedo.z, 1);
					}

					dst.geominfo.albedo[pos] = path.albedo;
				}
			}
		}
	}
}
