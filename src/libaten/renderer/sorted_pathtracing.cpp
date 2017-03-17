#include "renderer/sorted_pathtracing.h"
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

				auto& path = paths[pos];

				if (!path.isTerminate) {
					if (path.sampler) {
						auto rnd = (Sobol*)path.sampler->getRandom();
						rnd->reset((y * height * 4 + x * 4) * m_samples + sample + 1);
					}
					else {
						auto rnd = new Sobol((y * height * 4 + x * 4) * m_samples + sample + 1);
						auto sampler = new UniformDistributionSampler(rnd);
						path.sampler = sampler;
					}

					sampler* sampler = path.sampler;

					real u = real(x + sampler->nextSample()) / real(width);
					real v = real(y + sampler->nextSample()) / real(height);

					path.camsample = camera->sample(u, v, sampler);
					path.ray = path.camsample.r;

					path.x = x;
					path.y = y;

					path.isAlive = true;
					path.needWrite = true;

					path.contrib = vec3(0);
					path.throughput = vec3(1);
				}
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
				// ‰Šú‰».
				path.rec = hitrecord();
				path.isHit = scene->hit(path.ray, AT_MATH_EPSILON, AT_MATH_INF, path.rec);
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
		scene* scene,
		int depth,
		Path* paths,
		int numPath,
		vec4* dst)
	{
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int i = 0; i < numPath; i++) {
			auto& path = paths[i];

			if (path.isAlive && !path.isHit) {
				PathTracing::shadeMiss(scene, depth, path);

				auto idx = path.y * m_width + path.x;
				dst[idx] += vec4(path.contrib, 1);

				path.isAlive = false;
				path.needWrite = false;
			}
		}
	}

	void SortedPathTracing::shade(
		uint32_t sample,
		uint32_t depth,
		Path* paths,
		uint32_t* hitIds,
		int numHit,
		camera* cam,
		scene* scene)
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

			bool willContinue = PathTracing::shade(sampler, scene, cam, depth, path);

			if (!willContinue) {
				path.isAlive = false;
			}
		}
	}

	void SortedPathTracing::gather(
		Path* paths,
		int numPath,
		vec4* dst)
	{
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int i = 0; i < numPath; i++) {
			auto& path = paths[i];

			if (path.needWrite) {
				dst[i] += vec4(path.contrib, 1);

				path.needWrite = false;
			}
		}
	}

	void SortedPathTracing::render(
		Destination& dst,
		scene* scene,
		camera* camera)
	{
		m_width = dst.width;
		m_height = dst.height;
		vec4* color = dst.buffer;

		// TODO
		memset(color, 0, sizeof(vec4) * m_width * m_height);

		m_samples = dst.sample;
		m_maxDepth = dst.maxDepth;
		m_rrDepth = dst.russianRouletteDepth;

		if (m_rrDepth > m_maxDepth) {
			m_rrDepth = m_maxDepth - 1;
		}

		std::vector<Path> paths(m_width * m_height);
		std::vector<uint32_t> hitIds(m_width * m_height);

		for (uint32_t i = 0; i < m_samples; i++) {
			makePaths(
				m_width, m_height, i,
				&paths[0],
				camera);

			uint32_t depth = 0;

			while (depth < m_maxDepth) {
				hitPaths(
					&paths[0],
					(int)paths.size(),
					scene);

				auto numHit = compactionPaths(
					&paths[0],
					(int)paths.size(),
					&hitIds[0]);

				shadeMiss(
					scene, depth,
					&paths[0], (int)paths.size(), color);

				if (numHit == 0) {
					break;
				}

				shade(
					i,
					depth,
					&paths[0],
					&hitIds[0],
					numHit,
					camera,
					scene);

				depth++;
			}

			gather(&paths[0], (int)paths.size(), color);
		}

		if (m_samples > 1) {
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
			for (int y = 0; y < m_height; y++) {
				for (int x = 0; x < m_width; x++) {
					auto pos = y * m_width + x;

					auto& clr = color[pos];

					clr.r /= clr.w;
					clr.g /= clr.w;
					clr.b /= clr.w;
					clr.w = 1;
				}
			}
		}
	}
}
