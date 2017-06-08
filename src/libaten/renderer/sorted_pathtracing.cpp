#include "renderer/sorted_pathtracing.h"
#include "misc/thread.h"
#include "misc/timer.h"
#include "sampler/xorshift.h"
#include "sampler/halton.h"
#include "sampler/sobolproxy.h"

namespace aten
{
	void SortedPathTracing::makePaths(
		int width, int height,
		int sample,
		Path* paths,
		ray* rays,
		camera* camera)
	{

		auto time = timer::getSystemTime();

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				uint32_t idx = y * width + x;

				auto& path = paths[idx];

				if (!path.isTerminate) {
					if (path.sampler) {
						path.sampler->init((y * height * 4 + x * 4) * m_samples + sample + 1 + time.milliSeconds);
						//path.sampler->init((y * height * 4 + x * 4) * m_samples + sample + 1);
					}
					else {
						path.sampler = new Sobol((y * height * 4 + x * 4) * m_samples + sample + 1 + time.milliSeconds);
						//path.sampler = new XorShift((y * height * 4 + x * 4) * m_samples + sample + 1);
					}

					sampler* sampler = path.sampler;

					real u = real(x + sampler->nextSample()) / real(width);
					real v = real(y + sampler->nextSample()) / real(height);

					path.camsample = camera->sample(u, v, sampler);
					path.camSensitivity = camera->getSensitivity(
						path.camsample.posOnImageSensor,
						path.camsample.posOnLens);

					rays[idx] = path.camsample.r;

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
		const ray* rays,
		int numPath,
		scene* scene)
	{
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int i = 0; i < numPath; i++) {
			auto& path = paths[i];
			const auto& ray = rays[i];

			path.isHit = false;

			if (path.isAlive) {
				// 初期化.
				path.rec = hitrecord();
				path.ray = ray;

				Intersection isect;
				path.isHit = scene->hit(ray, AT_MATH_EPSILON, AT_MATH_INF, path.rec, isect);
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
				dst[idx] += path.camSensitivity * vec4(path.contrib, 1);

				path.isAlive = false;
				path.needWrite = false;
			}
		}
	}

	void SortedPathTracing::shade(
		uint32_t sample,
		uint32_t depth,
		Path* paths,
		ray* rays,
		ShadowRay* shadowRays,
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

			shadowRays[idx].isActive = false;

			auto& path = paths[idx];

			if (!path.isAlive) {
				continue;
			}

			auto sampler = path.sampler;
			bool willContinue = true;

			uint32_t rrDepth = m_rrDepth;

			// 交差位置の法線.
			// 物体からのレイの入出を考慮.
			vec3 orienting_normal = dot(path.rec.normal, path.ray.dir) < 0.0 ? path.rec.normal : -path.rec.normal;

			auto mtrl = material::getMaterial(path.rec.mtrlid);

			// Apply normal map.
			mtrl->applyNormalMap(orienting_normal, orienting_normal, path.rec.u, path.rec.v);

			// Implicit conection to light.
			if (mtrl->isEmissive()) {
				if (depth == 0) {
					// Ray hits the light directly.
					path.contrib = mtrl->color();
					path.isTerminate = true;
					willContinue = false;
				}
				else if (path.prevMtrl && path.prevMtrl->isSingular()) {
					auto emit = mtrl->color();
					path.contrib += path.throughput * emit;
					willContinue = false;
				}
				else {
					auto cosLight = dot(orienting_normal, -path.ray.dir);
					auto dist2 = squared_length(path.rec.p - path.ray.org);

					if (cosLight >= 0) {
						auto pdfLight = 1 / path.rec.area;

						// Convert pdf area to sradian.
						// http://www.slideshare.net/h013/edubpt-v100
						// p31 - p35
						pdfLight = pdfLight * dist2 / cosLight;

						auto misW = path.pdfb / (pdfLight + path.pdfb);

						auto emit = mtrl->color();

						path.contrib += path.throughput * misW * emit;

						// When ray hit the light, tracing will finish.
						willContinue = false;
					}
				}
			}

			if (!willContinue) {
				path.isAlive = false;
				continue;
			}

			// Explicit conection to light.
			if (!mtrl->isSingular())
			{
				real lightSelectPdf = 1;
				LightSampleResult sampleres;

				auto light = scene->sampleLight(
					path.rec.p,
					orienting_normal,
					sampler,
					lightSelectPdf, sampleres);

				if (light) {
					const vec3& posLight = sampleres.pos;
					const vec3& nmlLight = sampleres.nml;
					real pdfLight = sampleres.pdf;

					auto lightobj = sampleres.obj;

					vec3 dirToLight = normalize(sampleres.dir);
					shadowRays[idx] = ShadowRay(path.rec.p, dirToLight);

					auto cosShadow = dot(orienting_normal, dirToLight);

					auto bsdf = mtrl->bsdf(orienting_normal, path.ray.dir, dirToLight, path.rec.u, path.rec.v);
					auto pdfb = mtrl->pdf(orienting_normal, path.ray.dir, dirToLight, path.rec.u, path.rec.v);

					bsdf *= path.throughput;

					// Get light color.
					auto emit = sampleres.finalColor;

					path.lightPos = posLight;
					path.targetLight = light;
					path.lightcontrib = vec3(0);

					if (light->isSingular() || light->isInfinite()) {
						if (pdfLight > real(0)) {
							// TODO
							// ジオメトリタームの扱いについて.
							// singular light の場合は、finalColor に距離の除算が含まれている.
							// inifinite light の場合は、無限遠方になり、pdfLightに含まれる距離成分と打ち消しあう？.
							// （打ち消しあうので、pdfLightには距離成分は含んでいない）.
							auto misW = pdfLight / (pdfb + pdfLight);
							path.lightcontrib  = (misW * bsdf * emit * cosShadow / pdfLight) / lightSelectPdf;
						}
					}
					else {
						auto cosLight = dot(nmlLight, -dirToLight);

						if (cosShadow >= 0 && cosLight >= 0) {
							auto dist2 = squared_length(sampleres.dir);
							auto G = cosShadow * cosLight / dist2;

							if (pdfb > real(0) && pdfLight > real(0)) {
								// Convert pdf from steradian to area.
								// http://www.slideshare.net/h013/edubpt-v100
								// p31 - p35
								pdfb = pdfb * cosLight / dist2;

								auto misW = pdfLight / (pdfb + pdfLight);

								path.lightcontrib = (misW * (bsdf * emit * G) / pdfLight) / lightSelectPdf;
							}
						}
					}
				}
			}

			real russianProb = real(1);

			if (depth > rrDepth) {
				auto t = normalize(path.throughput);
				auto p = std::max(t.r, std::max(t.g, t.b));

				russianProb = sampler->nextSample();

				if (russianProb >= p) {
					path.contrib = vec3();
					willContinue = false;
				}
				else {
					russianProb = p;
				}
			}

			auto sampling = mtrl->sample(path.ray, orienting_normal, path.rec.normal, sampler, path.rec.u, path.rec.v);

			auto nextDir = normalize(sampling.dir);
			auto pdfb = sampling.pdf;
			auto bsdf = sampling.bsdf;

			real c = 1;
			if (!mtrl->isSingular()) {
				// TODO
				// AMDのはabsしているが....
				//c = aten::abs(dot(orienting_normal, nextDir));
				c = dot(orienting_normal, nextDir);
			}

			if (pdfb > 0 && c > 0) {
				path.throughput *= bsdf * c / pdfb;
				path.throughput /= russianProb;
			}
			else {
				willContinue = false;
			}

			path.prevMtrl = mtrl;

			path.pdfb = pdfb;

			// Make next ray.
			rays[idx] = aten::ray(path.rec.p, nextDir);

			if (!willContinue) {
				path.isAlive = false;
			}
		}
	}

	void SortedPathTracing::hitShadowRays(
		const Path* paths,
		ShadowRay* shadowrays,
		int numRay,
		scene* scene)
	{
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int i = 0; i < numRay; i++) {
			auto& shadowRay = shadowrays[i];

			if (shadowRay.isActive) {
				const auto& path = paths[i];

				hitrecord rec;
				shadowRay.isActive = scene->hitLight(
					path.targetLight,
					path.lightPos,
					shadowRay,
					AT_MATH_EPSILON, AT_MATH_INF, rec);
			}
		}
	}

	void SortedPathTracing::evalExplicitLight(
		Path* paths,
		const ShadowRay* shadowRays,
		uint32_t* hitIds,
		int numHit)
	{
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int i = 0; i < numHit; i++) {
			auto idx = hitIds[i];

			auto& path = paths[idx];

			const auto& shadowRay = shadowRays[idx];

			if (shadowRay.isActive) {
				path.contrib += path.lightcontrib;
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
				dst[i] += path.camSensitivity * vec4(path.contrib, 1);

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

		if (m_tmpbuffer.size() == 0) {
			m_tmpbuffer.resize(m_width * m_height);
		}
		
		vec4* color = &m_tmpbuffer[0];
		memset(color, 0, m_tmpbuffer.size() * sizeof(vec4));

		m_samples = dst.sample;
		m_maxDepth = dst.maxDepth;
		m_rrDepth = dst.russianRouletteDepth;

		if (m_rrDepth > m_maxDepth) {
			m_rrDepth = m_maxDepth - 1;
		}

		std::vector<Path> paths(m_width * m_height);
		std::vector<ray> rays(m_width * m_height);
		std::vector<ShadowRay> shadowRays(m_width * m_height);
		std::vector<uint32_t> hitIds(m_width * m_height);

		for (uint32_t i = 0; i < m_samples; i++) {
			makePaths(
				m_width, m_height, i,
				&paths[0],
				&rays[0],
				camera);

			uint32_t depth = 0;

			while (depth < m_maxDepth) {
				hitPaths(
					&paths[0],
					&rays[0],
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
					&rays[0],
					&shadowRays[0],
					&hitIds[0],
					numHit,
					camera,
					scene);

				hitShadowRays(&paths[0], &shadowRays[0], shadowRays.size(), scene);

				evalExplicitLight(
					&paths[0],
					&shadowRays[0],
					&hitIds[0],
					numHit);

				depth++;
			}

			gather(&paths[0], (int)paths.size(), color);
		}

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

				dst.buffer->put(x, y, clr);
			}
		}
	}
}
