#include "pathtracing.h"
#include "misc/thread.h"
#include "sampler/xorshift.h"
#include "sampler/halton.h"
#include "sampler/sobolproxy.h"
#include "sampler/UniformDistributionSampler.h"

namespace aten
{
	void SortedPathTracing::makePaths(
		int width, int height,
		int sample,
		Path* paths,
		camera* camera)
	{
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				uint32_t pos = y * width + x;
				if (paths[pos].sampler) {
					auto rnd = (Sobol*)paths[pos].sampler->getRandom();
					rnd->reset((y * height * 4 + x * 4) * m_samples + sample + 1);
				}
				else {
					auto rnd = new Sobol((y * height * 4 + x * 4) * m_samples + sample + 1);
					auto sampler = new UniformDistributionSampler(rnd);
					paths[pos].sampler = sampler;
				}

				sampler* sampler = paths[pos].sampler;

				real u = real(x + sampler->nextSample()) / real(width);
				real v = real(y + sampler->nextSample()) / real(height);

				paths[pos].camsample = camera->sample(u, v, sampler);
				paths[pos].r = paths[pos].camsample.r;

				paths[pos].x = x;
				paths[pos].y = y;

				paths[pos].isAlive = true;

				paths[pos].throughput = vec3(1, 1, 1);
			}
		}
	}

	void SortedPathTracing::hitPaths(
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

			if (path.isAlive) {
				path.isHit = scene->hit(path.r, AT_MATH_EPSILON, AT_MATH_INF, path.rec);
			}
		}
	}

	int SortedPathTracing::compactionPaths(
		Path* paths,
		int numPath,
		uint32_t* hitIds)
	{
		int cnt = 0;

		for (int i = 0; i < numPath; i++) {
			const auto& path = paths[i];
			if (path.isAlive && path.isHit) {
				hitIds[cnt++] = i;
			}
		}

		return cnt;
	}

	void SortedPathTracing::shadeMiss(
		Path* paths,
		int numPath,
		vec3* dst)
	{
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int i = 0; i < numPath; i++) {
			auto& path = paths[i];
			if (path.isAlive && !path.isHit) {
				auto bg = sampleBG(path.r);
				dst[i] += path.throughput * bg;
				path.isAlive = false;
			}
		}
	}

	void SortedPathTracing::shade(
		uint32_t depth,
		Path* paths,
		uint32_t* hitIds,
		int numHit,
		camera* cam,
		scene* scene,
		vec3* dst)
	{
		uint32_t rrDepth = m_rrDepth;

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int i = 0; i < numHit; i++) {
			auto idx = hitIds[i];
			auto& path = paths[idx];

			if (!path.isAlive) {
				continue;
			}

			auto sampler = path.sampler;

			const auto& ray = path.r;
			const auto& rec = path.rec;

			// 交差位置の法線.
			// 物体からのレイの入出を考慮.
			const vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

			// Implicit conection to light.
			if (rec.mtrl->isEmissive()) {
				path.isAlive = false;

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
					path.isAlive = false;
				}
				else {
					russianProb = p;
				}
			}

			auto sampling = rec.mtrl->sample(ray.dir, orienting_normal, rec, sampler, rec.u, rec.v);

			auto nextDir = normalize(sampling.dir);
			auto pdfb = sampling.pdf;
			auto bsdf = sampling.bsdf;

			real c = 1;
			if (!rec.mtrl->isSingular()) {
				// TODO
				// AMDのはabsしているが....
				c = dot(orienting_normal, nextDir);
			}

			if (pdfb > 0 && c > 0) {
				path.throughput *= bsdf * c / pdfb;
				path.throughput /= russianProb;
			}
			else {
				path.isAlive = false;
			}

			// Make next ray.
			path.r = aten::ray(rec.p + AT_MATH_EPSILON * nextDir, nextDir);
		}
	}

	void SortedPathTracing::render(
		Destination& dst,
		scene* scene,
		camera* camera)
	{
		int width = dst.width;
		int height = dst.height;
		vec3* color = dst.buffer;

		// TODO
		memset(color, 0, sizeof(vec3) * width * height);

		m_samples = dst.sample;
		m_maxDepth = dst.maxDepth;
		m_rrDepth = dst.russianRouletteDepth;

		if (m_rrDepth > m_maxDepth) {
			m_rrDepth = m_maxDepth - 1;
		}

		std::vector<Path> paths(width * height);
		std::vector<uint32_t> hitIds(width * height);

		for (int i = 0; i < m_samples; i++) {
			makePaths(
				width, height, i,
				&paths[0],
				camera);

			uint32_t depth = 0;

			while (depth < m_maxDepth) {
				hitPaths(
					&paths[0],
					paths.size(),
					scene);

				auto numHit = compactionPaths(
					&paths[0],
					paths.size(),
					&hitIds[0]);

				shadeMiss(&paths[0], paths.size(), color);

				if (numHit == 0) {
					break;
				}

				shade(
					depth,
					&paths[0],
					&hitIds[0],
					numHit,
					camera,
					scene,
					color);

				depth++;
			}
		}

		if (m_samples > 1) {
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					auto pos = y * width + x;
					color[pos] /= m_samples;
				}
			}
		}
	}
}
