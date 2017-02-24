#include "pathtracing.h"
#include "misc/thread.h"
#include "sampler/xorshift.h"
#include "sampler/halton.h"
#include "sampler/sobolproxy.h"
#include "sampler/UniformDistributionSampler.h"

namespace aten
{
	void PathTracingOpt::makePaths(
		int width, int height,
		Path* paths,
		camera* camera)
	{
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				auto rnd = new Sobol((y * height * 4 + x * 4));
				auto sampler = new UniformDistributionSampler(rnd);

				uint32_t pos = y * width + x;

				real u = real(x + sampler->nextSample()) / real(width);
				real v = real(y + sampler->nextSample()) / real(height);

				paths[pos].camsample = camera->sample(u, v, sampler);
				paths[pos].r = paths[pos].camsample.r;

				paths[pos].x = x;
				paths[pos].y = y;

				// TODO
				paths[pos].sampler = sampler;

				paths[pos].isValid = true;
			}
		}
	}

	void PathTracingOpt::hitPaths(
		Path* paths,
		int numPath,
		scene* scene)
	{
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int i = 0; i < numPath; i++) {
			auto& path = paths[i];

			path.isHit = false;

			if (path.isValid) {
				path.isHit = scene->hit(path.r, AT_MATH_EPSILON, AT_MATH_INF, path.rec);
			}
		}
	}

	int PathTracingOpt::compactionPaths(
		Path* paths,
		int numPath,
		uint32_t* hitIds)
	{
		int cnt = 0;

		for (int i = 0; i < numPath; i++) {
			const auto& path = paths[i];
			if (path.isHit) {
				hitIds[cnt++] = i;
			}
		}

		return cnt;
	}

	void PathTracingOpt::shadeMiss(
		Path* paths,
		int numPath,
		vec3* dst)
	{
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int i = 0; i < numPath; i++) {
			auto& path = paths[i];
			if (!path.isHit) {
				auto bg = sampleBG(path.r);
				dst[i] += path.throughput * bg;
				path.isValid = false;
			}
		}
	}

	void PathTracingOpt::shade(
		uint32_t depth,
		Path* paths,
		uint32_t* hitIds,
		int hitNum,
		camera* cam,
		vec3* dst)
	{
		uint32_t rrDepth = m_rrDepth;

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int i = 0; i < hitNum; i++) {
			auto idx = hitIds[i];
			auto& path = paths[idx];

			auto sampler = path.sampler;

			const auto& ray = path.r;
			const auto& rec = path.rec;

			// 交差位置の法線.
			// 物体からのレイの入出を考慮.
			const vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

			// Implicit conection to light.
			if (rec.mtrl->isEmissive()) {
				path.isValid = false;

				if (depth == 0) {
					// Ray hits the light directly.
					auto emit = rec.mtrl->color();
					dst[idx] = emit;
				}
				else {
					auto emit = rec.mtrl->color();

					dst[idx] += path.throughput * emit;
				}

				// When ray hit the light, tracing will finish.
				break;
			}

			if (depth == 0) {
				auto areaPdf = cam->getPdfImageSensorArea(rec.p, orienting_normal);

				//throughput *= Wdash;
				path.throughput /= areaPdf;
			}

			real russianProb = real(1);

			if (depth > rrDepth) {
				auto t = normalize(path.throughput);
				auto p = std::max(t.r, std::max(t.g, t.b));

				russianProb = sampler->nextSample();

				if (russianProb >= p) {
					path.isValid = false;
				}
				else {
					russianProb = p;
				}
			}

			auto sampling = rec.mtrl->sample(ray.dir, orienting_normal, rec, sampler, rec.u, rec.v);

			auto nextDir = sampling.dir;
			auto pdfb = sampling.pdf;
			auto bsdf = sampling.bsdf;

			// TODO
			// AMDのはabsしているが、正しい?
			//auto c = dot(orienting_normal, nextDir);
			auto c = aten::abs(dot(orienting_normal, nextDir));

			if (pdfb > 0) {
				path.throughput *= bsdf * c / pdfb;
				path.throughput /= russianProb;
			}
			else {
				break;
			}

			// Make next ray.
			path.r = aten::ray(rec.p + AT_MATH_EPSILON * nextDir, nextDir);
		}
	}

	void PathTracingOpt::render(
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

		std::vector<Path> paths(width * height);
		std::vector<uint32_t> hitIds(width * height);

		makePaths(
			width, height,
			&paths[0],
			camera);

		uint32_t depth = 0;

		while (depth < m_maxDepth) {
			hitPaths(
				&paths[0],
				paths.size(),
				scene);

			auto hitNum = compactionPaths(
				&paths[0],
				paths.size(),
				&hitIds[0]);

			if (depth == 0) {
				shadeMiss(&paths[0], paths.size(), color);
			}

			shade(
				depth,
				&paths[0],
				&hitIds[0],
				hitNum,
				camera,
				color);

			depth++;
		}
	}
}
