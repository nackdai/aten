#include "renderer/svgf_pt.h"
#include "misc/thread.h"
#include "misc/timer.h"
#include "renderer/nonphotoreal.h"
#include "sampler/xorshift.h"
#include "sampler/halton.h"
#include "sampler/sobolproxy.h"
#include "sampler/wanghash.h"

//#define RELEASE_DEBUG

#ifdef RELEASE_DEBUG
#pragma optimize( "", off)
#endif

namespace aten
{
	// NOTE
	// https://www.slideshare.net/shocker_0x15/ss-52688052

	static inline bool isInvalidColor(const vec3& v)
	{
		bool b = isInvalid(v);
		if (!b) {
			if (v.x < 0 || v.y < 0 || v.z < 0) {
				b = true;
			}
		}

		return b;
	}

	SVGFPathTracing::Path SVGFPathTracing::radiance(
		int x, int y,
		int w, int h,
		sampler* sampler,
		const ray& inRay,
		camera* cam,
		CameraSampleResult& camsample,
		scene* scene)
	{
		return radiance(x, y, w, h, sampler, m_maxDepth, inRay, cam, camsample, scene);
	}

	SVGFPathTracing::Path SVGFPathTracing::radiance(
		int x, int y,
		int w, int h,
		sampler* sampler,
		uint32_t maxDepth,
		const ray& inRay,
		camera* cam,
		CameraSampleResult& camsample,
		scene* scene)
	{
		uint32_t depth = 0;
		uint32_t rrDepth = m_rrDepth;

		Path path;
		path.ray = inRay;
		path.idx = y * w + x;

		while (depth < maxDepth) {
			path.rec = hitrecord();

#if 1
			bool willContinue = true;
			Intersection isect;

			if (scene->hit(path.ray, AT_MATH_EPSILON, AT_MATH_INF, path.rec, isect)) {
				willContinue = shade(sampler, scene, cam, camsample, depth, path);
			}
			else {
				shadeMiss(scene, depth, path);
				willContinue = false;
			}

			if (!willContinue) {
				break;
			}
#else
			if (scene->hit(path.ray, AT_MATH_EPSILON, AT_MATH_INF, path.rec)) {
				bool willContinue = shade(sampler, scene, cam, depth, path);
				if (!willContinue) {
					break;
				}
			}
			else {
				shadeMiss(scene, depth, path);
				break;
			}
#endif

			depth++;
		}

		return std::move(path);
	}

	bool SVGFPathTracing::shade(
		sampler* sampler,
		scene* scene,
		camera* cam,
		CameraSampleResult& camsample,
		int depth,
		Path& path)
	{
		uint32_t rrDepth = m_rrDepth;

		// 交差位置の法線.
		// 物体からのレイの入出を考慮.
		vec3 orienting_normal = dot(path.rec.normal, path.ray.dir) < 0.0 ? path.rec.normal : -path.rec.normal;

		auto mtrl = material::getMaterial(path.rec.mtrlid);

		// Apply normal map.
		mtrl->applyNormalMap(orienting_normal, orienting_normal, path.rec.u, path.rec.v);

		if (depth == 0) {
			auto& aov = getCurAov();

			aov[path.idx].normal = orienting_normal;
			aov[path.idx].mtrlid = path.rec.mtrlid;

			auto& camParam = const_cast<aten::CameraParameter&>(cam->param());
			camParam.znear = real(0.1);
			camParam.zfar = real(10000.0);

			aten::mat4 mtxW2V;
			mtxW2V.lookat(
				camParam.origin,
				camParam.center,
				camParam.up);

			m_mtxV2C.perspective(
				camParam.znear,
				camParam.zfar,
				camParam.vfov,
				camParam.aspect);

			m_mtxC2V = m_mtxV2C;
			m_mtxC2V.invert();

			aten::mat4 mtxW2C = m_mtxV2C * mtxW2V;

			aten::vec4 pos = aten::vec4(path.rec.p, 1);
			pos = mtxW2C.apply(pos);

			aov[path.idx].depth = pos.w;
		}

		// Implicit conection to light.
		if (mtrl->isEmissive()) {
			if (depth == 0) {
				// Ray hits the light directly.
				path.contrib = mtrl->color();
				path.isTerminate = true;
				return false;
			}
			else if (path.prevMtrl && path.prevMtrl->isSingular()) {
				auto emit = mtrl->color();
				path.contrib += path.throughput * emit;
				return false;
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
					return false;
				}
			}
		}

		// Non-Photo-Real.
		if (mtrl->isNPR()) {
			path.contrib = shadeNPR(mtrl, path.rec.p, orienting_normal, path.rec.u, path.rec.v, scene, sampler);
			path.isTerminate = true;
			return false;
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
				aten::ray shadowRay(path.rec.p, dirToLight);

				hitrecord tmpRec;

				if (scene->hitLight(light, posLight, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, tmpRec)) {
					// Shadow ray hits the light.
					auto cosShadow = dot(orienting_normal, dirToLight);

					auto bsdf = mtrl->bsdf(orienting_normal, path.ray.dir, dirToLight, path.rec.u, path.rec.v);
					auto pdfb = mtrl->pdf(orienting_normal, path.ray.dir, dirToLight, path.rec.u, path.rec.v);

					bsdf *= path.throughput;

					// Get light color.
					auto emit = sampleres.finalColor;

					if (light->isSingular() || light->isInfinite()) {
						if (pdfLight > real(0) && cosShadow >= 0) {
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

								path.contrib += (misW * (bsdf * emit * G) / pdfLight) / lightSelectPdf;
							}
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
				return false;
			}
			else {
				russianProb = p;
			}
		}

		auto sampling = mtrl->sample(path.ray, orienting_normal, path.rec.normal, sampler, path.rec.u, path.rec.v);

		auto nextDir = normalize(sampling.dir);
		auto pdfb = sampling.pdf;
		auto bsdf = sampling.bsdf;

#if 1
		real c = 1;
		if (!mtrl->isSingular()) {
			// TODO
			// AMDのはabsしているが....
			//c = aten::abs(dot(orienting_normal, nextDir));
			c = dot(orienting_normal, nextDir);
		}
#else
		auto c = dot(orienting_normal, nextDir);
#endif

		//if (pdfb > 0) {
		if (pdfb > 0 && c > 0) {
			path.throughput *= bsdf * c / pdfb;
			path.throughput /= russianProb;
		}
		else {
			return false;
		}

		path.prevMtrl = mtrl;

		path.pdfb = pdfb;

		// Make next ray.
		path.ray = aten::ray(path.rec.p, nextDir);

		return true;
	}

	void SVGFPathTracing::shadeMiss(
		scene* scene,
		int depth,
		Path& path)
	{
		auto ibl = scene->getIBL();
		if (ibl) {
			if (depth == 0) {
				auto bg = ibl->getEnvMap()->sample(path.ray);
				path.contrib += path.throughput * bg;
				path.isTerminate = true;
			}
			else {
				auto pdfLight = ibl->samplePdf(path.ray);
				auto misW = path.pdfb / (pdfLight + path.pdfb);
				auto emit = ibl->getEnvMap()->sample(path.ray);
				path.contrib += path.throughput * misW * emit;
			}
		}
		else {
			auto bg = sampleBG(path.ray);
			path.contrib += path.throughput * bg;
		}
	}

	inline void computePrevScreenPos(
		int ix, int iy,
		float centerDepth,
		int width, int height,
		aten::vec4* prevPos,
		const aten::mat4* mtxs)
	{
		// NOTE
		// Pview = (Xview, Yview, Zview, 1)
		// mtxV2C = W 0 0  0
		//          0 H 0  0
		//          0 0 A  B
		//          0 0 -1 0
		// mtxV2C * Pview = (Xclip, Yclip, Zclip, Wclip) = (Xclip, Yclip, Zclip, Zview)
		//  Wclip = Zview = depth
		// Xscr = Xclip / Wclip = Xclip / Zview = Xclip / depth
		// Yscr = Yclip / Wclip = Yclip / Zview = Yclip / depth
		//
		// Xscr * depth = Xclip
		// Xview = mtxC2V * Xclip

		const aten::mat4 mtxC2V = mtxs[0];
		const aten::mat4 mtxPrevV2C = mtxs[1];

		aten::vec3 uv(ix + 0.5, iy + 0.5, 1);
		uv /= aten::vec3(width - 1, height - 1, 1);	// [0, 1]
		uv = uv * 2.0f - 1.0f;	// [0, 1] -> [-1, 1]

		aten::vec4 pos(uv.x, uv.y, 0, 0);

		// Screen-space -> Clip-space.
		pos.x *= centerDepth;
		pos.y *= centerDepth;

		// Clip-space -> View-space
		pos = mtxC2V.apply(pos);
		pos.z = -centerDepth;
		pos.w = 1.0;

		// Reproject previous screen position.
		*prevPos = mtxPrevV2C.apply(pos);
		*prevPos /= prevPos->w;

		*prevPos = *prevPos * 0.5 + 0.5;	// [-1, 1] -> [0, 1]
	}
	
	void SVGFPathTracing::temporalReprojection(
		Destination& dst, 
		camera* camera,
		int width, int height)
	{
#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
//#pragma omp parallel for
#endif
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				temporalReprojection(dst, camera, x, y, width, height);
			}
		}
	}

#pragma optimize( "", off)

	void SVGFPathTracing::temporalReprojection(
		Destination& dst,
		camera* camera,
		int ix, int iy,
		int width, int height)
	{
		const auto& cam = camera->param();

		int idx = iy * width + ix;

		auto& curAov = getCurAov()[idx];

		const float centerDepth = aten::clamp(curAov.depth, cam.znear, cam.zfar);
		const int centerMeshId = curAov.mtrlid;

		auto curColor = curAov.color;

		auto lum0 = aten::color::luminance(curColor);

		const auto centerNormal = curAov.normal;

		aten::mat4 mtxs[] = {
			m_mtxC2V,
			m_mtxPrevV2C,
		};

		aten::vec4 sum(0, 0, 0, 0);
		float weight = 0.0f;

		static const float zThreshold = 0.05f;
		static const float nThreshold = 0.98f;
		static const float lThreshold = 0.6f;

		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				int xx = clamp(ix + x, 0, width - 1);
				int yy = clamp(iy + y, 0, height - 1);

				// 前のフレームのクリップ空間座標を計算.
				aten::vec4 prevPos;
				computePrevScreenPos(
					xx, yy,
					centerDepth,
					width, height,
					&prevPos,
					mtxs);

				// [0, 1]の範囲内に入っているか.
				bool isInsideX = (0.0 <= prevPos.x) && (prevPos.x <= 1.0);
				bool isInsideY = (0.0 <= prevPos.y) && (prevPos.y <= 1.0);

				if (isInsideX && isInsideY) {
					// 前のフレームのスクリーン座標.
					int px = (int)(prevPos.x * width - 0.5f);
					int py = (int)(prevPos.y * height - 0.5f);

					px = clamp(px, 0, width - 1);
					py = clamp(py, 0, height - 1);

					int pidx = py * width + px;

					const auto& prevAov = getPrevAov()[pidx];

					const float prevDepth = aten::clamp(prevAov.depth, cam.znear, cam.zfar);
					const int prevMeshId = prevAov.mtrlid;
					const auto prevNormal = prevAov.normal;

					auto prev = prevAov.color;

					auto lum1 = aten::color::luminance(prev);

					float Wz = clamp((zThreshold - abs(1 - centerDepth / prevDepth)) / zThreshold, 0.0f, 1.0f);
					float Wn = clamp((dot(centerNormal, prevNormal) - nThreshold) / (1.0f - nThreshold), 0.0f, 1.0f);
					float Wm = centerMeshId == prevMeshId ? 1.0f : 0.0f;

					float W = Wz * Wn * Wm;
					sum += prev * W;
					weight += W;
				}
			}
		}

		if (weight > 0.0f) {
			sum /= weight;
			curColor = 0.2 * curColor + 0.8 * sum;
		}

		curColor.w = 1;

		curAov.color = curColor;

		//dst.buffer->put(ix, iy, curColor);

		if (ix == 257 && iy == 512 - 56) {
			int xxx = 0;
		}

		{
			auto centerMoment = curAov.moments;

			// 前のフレームのクリップ空間座標を計算.
			aten::vec4 prevPos;
			computePrevScreenPos(
				ix, iy,
				centerDepth,
				width, height,
				&prevPos,
				mtxs);

			// [0, 1]の範囲内に入っているか.
			bool isInsideX = (0.0 <= prevPos.x) && (prevPos.x <= 1.0);
			bool isInsideY = (0.0 <= prevPos.y) && (prevPos.y <= 1.0);

			// 積算フレーム数のリセット.
			int frame = 1;

			if (isInsideX && isInsideY) {
				// 前のフレームのスクリーン座標.
				int px = (int)(prevPos.x * width - 0.5f);
				int py = (int)(prevPos.y * height - 0.5f);

				px = clamp(px, 0, width - 1);
				py = clamp(py, 0, height - 1);

				int pidx = py * width + px;

				const auto& prevAov = getPrevAov()[pidx];

				const float prevDepth = aten::clamp(prevAov.depth, cam.znear, cam.zfar);
				const int prevMeshId = prevAov.mtrlid;
				const auto prevNormal = prevAov.normal;

				if (abs(1 - centerDepth / prevDepth) < zThreshold
					&& dot(centerNormal, prevNormal) > nThreshold
					&& centerMeshId == prevMeshId)
				{
					auto prevMoment = prevAov.moments;

					//centerMoment += prevMoment;
					centerMoment += prevMoment;

					// 積算フレーム数を１増やす.
					frame = (int)prevMoment.w + 1;
				}
			}

			centerMoment.w = frame;

			curAov.moments = centerMoment;
		}
	}

	void SVGFPathTracing::estimateVariance(
		Destination& dst,
		int width, int height)
	{
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				estimateVariance(dst, x, y, width, height);
			}
		}
	}

	const SVGFPathTracing::AOV& SVGFPathTracing::sampleAov(
		int ix, int iy,
		int width, int height)
	{
		ix = clamp(ix, 0, width - 1);
		iy = clamp(iy, 0, height - 1);

		int idx = iy * width + ix;

		auto& aov = getCurAov()[idx];

		return aov;
	}

	void SVGFPathTracing::estimateVariance(
		Destination& dst,
		int ix, int iy,
		int width, int height)
	{
		if (ix == 257 && iy == 512 - 56) {
			int xxx = 0;
		}

		int idx = iy * width + ix;

		auto& curAov = getCurAov()[idx];

		auto centerMoment = curAov.moments;

		int frame = (int)centerMoment.w;

		centerMoment /= centerMoment.w;

		// 分散を計算.
		float var = centerMoment.x - centerMoment.y * centerMoment.y;

		if (frame < 4) {
			// 積算フレーム数が４未満 or Disoccludedされている.
			// 7x7birateral filterで輝度を計算.

			static const int radius = 3;
			static const float sigmaN = 0.005f;
			static const float sigmaD = 0.005f;
			static const float sigmaS = 8;

			auto centerNormal = curAov.normal;
			float centerDepth = curAov.depth;
			int centerMeshId = curAov.mtrlid;

			vec3 sum(0, 0, 0);
			float weight = 0.0f;

			for (int v = -radius; v <= radius; v++)
			{
				for (int u = -radius; u <= radius; u++)
				{
					auto saov = sampleAov(ix + u, iy + v, width, height);

					auto moment = saov.moments;
					auto sampleNml = saov.normal;
					auto sampleDepth = saov.depth;
					int sampleMeshId = saov.mtrlid;

					moment /= moment.w;

					float n = 1 - dot(sampleNml, centerNormal);
					float Wn = exp(-0.5f * n * n / (sigmaN * sigmaN));

					float d = 1 - std::min(centerDepth, sampleDepth) / std::max(centerDepth, sampleDepth);
					float Wd = exp(-0.5f * d * d / (sigmaD * sigmaD));

					float Ws = exp(-0.5f * (u * u + v * v) / (sigmaS * sigmaS));

					float Wm = centerMeshId == sampleMeshId ? 1.0f : 0.0f;

					float W = Ws * Wn * Wd * Wm;
					sum += (vec3)moment * W;
					weight += W;
				}
			}

			if (weight > 0.0f) {
				sum /= weight;
			}

			var = sum.x - sum.y * sum.y;
		}

		// TODO
		// 分散はマイナスにならないが・・・・
		var = abs(var);

		dst.buffer->put(ix, iy, vec4(var, var, var, 1));
	}

//#pragma optimize( "", on)

	void SVGFPathTracing::render(
		Destination& dst,
		scene* scene,
		camera* camera)
	{
		int width = dst.width;
		int height = dst.height;
		uint32_t samples = dst.sample;

		m_maxDepth = dst.maxDepth;
		m_rrDepth = dst.russianRouletteDepth;

		if (m_rrDepth > m_maxDepth) {
			m_rrDepth = m_maxDepth - 1;
		}

		for (int i = 0; i < 2; i++) {
			if (m_aov[i].empty()) {
				m_aov[i].resize(width * height);
			}
		}

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp parallel
#endif
		{
			auto idx = thread::getThreadIdx();

			//XorShift rnd(idx);
			//UniformDistributionSampler sampler(&rnd);

			auto time = timer::getSystemTime();

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int pos = y * width + x;

					auto& aov = getCurAov();

					aov[pos].normal = vec3(0, 0, 0);
					aov[pos].depth = -1;
					aov[pos].meshid = -1;
					aov[pos].meshid = -1;
					aov[pos].moments = vec4(0, 0, 0, 0);
					aov[pos].color = vec4(0, 0, 0, 0);

					vec3 col = vec3(0);
					uint32_t cnt = 0;

					if (x == 228 && y == 471) {
						int xxx = 0;
					}

					for (uint32_t i = 0; i < samples; i++) {
						int seed = (y * height * 4 + x * 4) * samples + i + 1;
#ifndef RELEASE_DEBUG
						seed += time.milliSeconds;
#endif

						//XorShift rnd(seed);
						//Halton rnd(seed);
						//Sobol rnd(seed);
						WangHash rnd(seed);

						real u = real(x + rnd.nextSample()) / real(width);
						real v = real(y + rnd.nextSample()) / real(height);

						auto camsample = camera->sample(u, v, &rnd);

						auto ray = camsample.r;

						auto path = radiance(
							x, y,
							width, height,
							&rnd,
							ray, 
							camera,
							camsample,
							scene);

						if (isInvalidColor(path.contrib)) {
							AT_PRINTF("Invalid(%d/%d[%d])\n", x, y, i);
							continue;
						}

						auto pdfOnImageSensor = camsample.pdfOnImageSensor;
						auto pdfOnLens = camsample.pdfOnLens;

						auto s = camera->getSensitivity(
							camsample.posOnImageSensor,
							camsample.posOnLens);

						auto c = path.contrib * s / (pdfOnImageSensor * pdfOnLens);

						col += c;
						cnt++;

						if (path.isTerminate) {
							break;
						}
					}

					col /= (real)cnt;

					//dst.buffer->put(x, y, vec4(col, 1));

					aov[pos].color = vec4(col, 1);

					if (x == 254 && y == 453) {
						int xxx = 0;
					}

					auto lum = aten::color::luminance(col);
					aov[pos].moments.x = lum * lum;
					aov[pos].moments.y = lum;
					aov[pos].moments.z = 0;
					aov[pos].moments.w = 1;
				}
			}
		}

		if (m_isFirstRender) {

		}
		else {
			// Temporal Reprojection.
			temporalReprojection(dst, camera, width, height);
		}

		estimateVariance(dst, width, height);

		m_mtxPrevV2C = m_mtxV2C;

		m_curPos = 1 - m_curPos;

		m_isFirstRender = false;
	}
}
