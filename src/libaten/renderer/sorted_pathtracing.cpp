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
					}
					else {
						path.sampler = new Sobol((y * height * 4 + x * 4) * m_samples + sample + 1 + time.milliSeconds);
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

					path.contrib = make_float3(0);
					path.throughput = make_float3(1);
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

			if (path.isAlive && ray.isActive) {
				// 初期化.
				path.rec = hitrecord();
				path.isHit = scene->hit(ray, AT_MATH_EPSILON, AT_MATH_INF, path.rec);
			}
		}
	}

	void SortedPathTracing::hitRays(
		ray* rays,
		int numRay,
		scene* scene)
	{
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int i = 0; i < numRay; i++) {
			auto& ray = rays[i];

			if (ray.isActive) {
				hitrecord rec;
				ray.isActive = scene->hit(ray, AT_MATH_EPSILON, AT_MATH_INF, rec);
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
		ray* shadowRays,
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
			bool willContinue = false;

			uint32_t rrDepth = m_rrDepth;

			// 交差位置の法線.
			// 物体からのレイの入出を考慮.
			path.orienting_normal = dot(path.rec.normal, path.ray.dir) < 0.0 ? path.rec.normal : -path.rec.normal;

			// Implicit conection to light.
			if (path.rec.mtrl->isEmissive()) {
				if (depth == 0) {
					// Ray hits the light directly.
					path.contrib = path.rec.mtrl->color();
					path.isTerminate = true;
					willContinue = false;
				}
				else if (path.prevMtrl && path.prevMtrl->isSingular()) {
					auto emit = path.rec.mtrl->color();
					path.contrib += path.throughput * emit;
					willContinue = false;
				}
				else {
					auto cosLight = dot(path.orienting_normal, -path.ray.dir);
					auto dist2 = (path.rec.p - path.ray.org).squared_length();

					if (cosLight >= 0) {
						auto pdfLight = 1 / path.rec.area;

						// Convert pdf area to sradian.
						// http://www.slideshare.net/h013/edubpt-v100
						// p31 - p35
						pdfLight = pdfLight * dist2 / cosLight;

						auto misW = path.pdfb / (pdfLight + path.pdfb);

						auto emit = path.rec.mtrl->color();

						path.contrib += path.throughput * misW * emit;

						// When ray hit the light, tracing will finish.
						willContinue = false;
					}
				}
			}

			// Apply normal map.
			path.rec.mtrl->applyNormalMap(path.orienting_normal, path.orienting_normal, path.rec.u, path.rec.v);

			shadowRays[idx].isActive = false;

			// Explicit conection to light.
			if (!path.rec.mtrl->isSingular())
			{
				real lightSelectPdf = 1;
				LightSampleResult sampleres;

				auto light = scene->sampleLight(
					path.rec.p,
					path.orienting_normal,
					sampler,
					lightSelectPdf, sampleres);

				if (light) {
					const vec3& posLight = sampleres.pos;
					const vec3& nmlLight = sampleres.nml;
					real pdfLight = sampleres.pdf;

					auto lightobj = sampleres.obj;

					vec3 dirToLight = normalize(sampleres.dir);
					shadowRays[idx] = aten::ray(path.rec.p, dirToLight);

					path.pdfLight = sampleres.pdf;
					path.dist2ToLight = sampleres.dir.squared_length();
					path.cosLight = dot(nmlLight, -dirToLight);
					path.lightSelectPdf = lightSelectPdf;
					path.lightAttrib = light->param().attrib;
					path.lightColor = sampleres.finalColor;
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

			auto sampling = path.rec.mtrl->sample(path.ray, path.orienting_normal, path.rec.normal, sampler, path.rec.u, path.rec.v);

			auto nextDir = normalize(sampling.dir);
			auto pdfb = sampling.pdf;
			auto bsdf = sampling.bsdf;

			real c = 1;
			if (!path.rec.mtrl->isSingular()) {
				// TODO
				// AMDのはabsしているが....
				//c = aten::abs(dot(orienting_normal, nextDir));
				c = dot(path.orienting_normal, nextDir);
			}

			if (pdfb > 0 && c > 0) {
				path.throughput *= bsdf * c / pdfb;
				path.throughput /= russianProb;
			}
			else {
				willContinue = false;
			}

			path.prevMtrl = path.rec.mtrl;

			path.pdfb = pdfb;

			// Make next ray.
			rays[idx] = aten::ray(path.rec.p, nextDir);

			if (!willContinue) {
				path.isAlive = false;
			}
		}
	}

#pragma optimize( "", off)

	void SortedPathTracing::evalExplicitLight(
		Path* paths,
		const ray* shadowRays,
		uint32_t* hitIds,
		int numHit)
	{
#ifdef ENABLE_OMP
//#pragma omp parallel for
#endif
		for (int i = 0; i < numHit; i++) {
			auto idx = hitIds[i];
			auto& path = paths[idx];
			const ray& shadowRay = shadowRays[idx];

			if (shadowRay.isActive) {
				// Shadow ray hits the light.
				auto cosShadow = dot(path.orienting_normal, shadowRay.dir);

				auto bsdf = path.rec.mtrl->bsdf(path.orienting_normal, path.ray.dir, shadowRay.dir, path.rec.u, path.rec.v);
				auto pdfb = path.rec.mtrl->pdf(path.orienting_normal, path.ray.dir, shadowRay.dir, path.rec.u, path.rec.v);

				bsdf *= path.throughput;

				// Get light color.
				auto emit = path.lightColor;
				real pdfLight = path.pdfLight;
				real lightSelectPdf = path.lightSelectPdf;

				if (path.lightAttrib.isSingular || path.lightAttrib.isInfinite) {
					if (path.pdfLight > real(0)) {
						// TODO
						// ジオメトリタームの扱いについて.
						// singular light の場合は、finalColor に距離の除算が含まれている.
						// inifinite light の場合は、無限遠方になり、pdfLightに含まれる距離成分と打ち消しあう？.
						// （打ち消しあうので、pdfLightには距離成分は含んでいない）.
						auto misW = pdfLight / (pdfb + pdfLight);
						path.contrib += (misW * bsdf * emit * cosShadow / pdfLight) / lightSelectPdf;
					}
				}
				else {
					auto cosLight = path.cosLight;;

					if (cosShadow >= 0 && cosLight >= 0) {
						auto dist2 = path.dist2ToLight;
						auto G = cosShadow * cosLight / dist2;

						if (pdfb > real(0) && pdfLight > real(0)) {
							// Convert pdf from steradian to area.
							// http://www.slideshare.net/h013/edubpt-v100
							// p31 - p35
							pdfb = pdfb * cosLight / dist2;

							auto misW = pdfLight / (pdfb + pdfLight);

							path.contrib += (misW * (bsdf * emit * G) / pdfLight) / lightSelectPdf;
						}
					}
				}
			}
		}
	}
#pragma optimize( "", on)

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
		std::vector<ray> shadowRays(m_width * m_height);
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

				hitRays(&shadowRays[0], shadowRays.size(), scene);

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
