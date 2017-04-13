#include "renderer/bdpt.h"
#include "misc/thread.h"
#include "misc/timer.h"
#include "material/lambert.h"
#include "sampler/xorshift.h"
#include "sampler/halton.h"
#include "sampler/sobolproxy.h"
#include "sampler/UniformDistributionSampler.h"

#define BDPT_DEBUG

#ifdef BDPT_DEBUG
#pragma optimize( "", off) 
#endif

namespace aten
{
	// NOTE
	// http://www.ci.i.u-tokyo.ac.jp/~hachisuka/smallpssmlt.cpp

	vec3 BDPT::trace(
		bool isEyePath,
		std::vector<Vertex>& vs,
		uint32_t maxDepth,
		const ray& r,
		sampler* sampler,
		scene* scene,
		camera* camera)
	{
		uint32_t depth = 0;

		ray ray = r;

		vec3 throughput(1);
		vec3 contrib(0);

		while (depth < maxDepth) {
			hitrecord rec;
			bool isHit = scene->hit(ray, AT_MATH_EPSILON, AT_MATH_INF, rec);

			vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

			if (isEyePath) {
				if (!isHit) {
					break;
				}

				// For eye path.
				Vertex vtx(
					rec.p,
					orienting_normal,
					rec.obj,
					rec.mtrl,
					rec.u, rec.v);

				if (rec.mtrl->isEmissive()) {
					auto emit = rec.mtrl->color();
					contrib = throughput * emit;

					if (depth == 0) {
						// NOTE
						// Marker which ray hits light directly.
						vtx.x = 0;
						vtx.y = 0;
					}

					vs.push_back(vtx);

					break;
				}
				else {
					vs.push_back(vtx);
				}
			}
			else {
				// For light path.
				vec3 posOnImageSensor;
				vec3 posOnLens;
				vec3 posOnObjectPlane;
				int pixelx;
				int pixely;

				// レンズと交差判定.
				auto lens_t = camera->hitOnLens(
					ray,
					posOnLens,
					posOnObjectPlane,
					posOnImageSensor,
					pixelx, pixely);

				if (AT_MATH_EPSILON < lens_t && lens_t < rec.t) {
					// レイがレンズにヒット＆イメージセンサにヒット.
					const vec3& camnml = camera->getDir();

					Vertex vtx(
						rec.p,
						camnml,
						nullptr,
						nullptr,
						0, 0);

					vtx.x = aten::clamp(pixelx, 0, m_width);
					vtx.y = aten::clamp(pixely, 0, m_height);

					vs.push_back(vtx);

					contrib = throughput;

					break;
				}

				if (!isHit) {
					break;
				}

				vs.push_back(Vertex(
					rec.p,
					orienting_normal,
					rec.obj,
					rec.mtrl,
					rec.u, rec.v));
			}

			auto sampling = rec.mtrl->sample(ray, orienting_normal, rec, sampler, rec.u, rec.v);

			auto nextDir = normalize(sampling.dir);
			auto pdfb = sampling.pdf;
			auto bsdf = sampling.bsdf;

#if 1
			real c = 1;
			if (!rec.mtrl->isSingular()) {
				// TODO
				// AMDのはabsしているが....
				//c = aten::abs(dot(orienting_normal, nextDir));
				c = dot(orienting_normal, nextDir);
			}
#else
			auto c = dot(orienting_normal, nextDir);
#endif

			if (pdfb > 0 && c > 0) {
				throughput *= bsdf * c / pdfb;
			}

			ray = aten::ray(rec.p + nextDir * AT_MATH_EPSILON, nextDir);

			depth++;
		}

		return std::move(contrib);
	}

	BDPT::Result BDPT::genEyePath(
		std::vector<Vertex>& vs,
		int x, int y,
		sampler* sampler,
		scene* scene,
		camera* camera)
	{
		real u = real(x + sampler->nextSample()) / (real)m_width;
		real v = real(y + sampler->nextSample()) / (real)m_height;

		auto camsample = camera->sample(u, v, sampler);

		// カメラ.
		vs.push_back(Vertex(
			camsample.posOnLens,
			camsample.nmlOnLens,
			nullptr,
			nullptr,
			0, 0,
			x, y));

		auto contrib = trace(
			true,
			vs,
			m_maxDepth,
			camsample.r,
			sampler,
			scene,
			camera);

		Result res;

		res.contrib = contrib;
		res.camsample = camsample;

		const auto& endEye = vs[vs.size() - 1];
		if (endEye.x >= 0) {
			res.x = x;
			res.y = y;
		}

		return std::move(res);
	}

	BDPT::Result BDPT::genLightPath(
		std::vector<Vertex>& vs,
		Light* light,
		sampler* sampler,
		scene* scene,
		camera* camera)
	{
		// TODO
		// Only AreaLight...

		auto res = light->getSamplePosNormalPdf(sampler);
		auto pos = std::get<0>(res);
		auto nml = std::get<1>(res);
		auto pdf = std::get<2>(res);

		hitable* lightobj = const_cast<hitable*>(light->getLightObject());

		// 光源.
		vs.push_back(Vertex(
			pos,
			nml,
			lightobj,
			nullptr,
			light,
			pdf,
			0, 0));

		vec3 dir;
		{
			// normalの方向を基準とした正規直交基底(w, u, v)を作る.
			// この基底に対する半球内で次のレイを飛ばす.
			vec3 n, t, b;

			n = nml;

			// nと平行にならないようにする.
			if (fabs(n.x) > AT_MATH_EPSILON) {
				t = normalize(cross(vec3(0.0, 1.0, 0.0), n));
			}
			else {
				t = normalize(cross(vec3(1.0, 0.0, 0.0), n));
			}
			b = cross(n, t);

			// コサイン項を使った重点的サンプリング.
			const real r1 = 2 * AT_MATH_PI * sampler->nextSample();
			const real r2 = sampler->nextSample();
			const real r2s = sqrt(r2);

			const real x = aten::cos(r1) * r2s;
			const real y = aten::sin(r1) * r2s;
			const real z = aten::sqrt(real(1) - r2);

			dir = normalize((t * x + b * y + n * z));
		}

		ray r(pos + dir * AT_MATH_EPSILON, dir);

		vec3 contrib = trace(
			false,
			vs,
			m_maxDepth,
			r,
			sampler,
			scene,
			camera);

		contrib *= light->getLe();

		Result lightRes;

		lightRes.contrib = contrib;

		const auto& endLight = vs[vs.size() - 1];
		if (endLight.x >= 0) {
			lightRes.x = endLight.x;
			lightRes.y = endLight.y;
		}

		return std::move(lightRes);
	}

	bool BDPT::isConnectable(
		scene* scene,
		camera* camera,
		const std::vector<Vertex>& eyepath,
		const int numEye,
		const std::vector<Vertex>& lightpath,
		const int numLight,
		int& px, int& py)
	{
		vec3 firstRayDir; // Direction from camera origin.
		vec3 eyeOrg;

		bool result = false;

		px = -1;
		py = -1;

		if ((numEye == 0) && (numLight >= 2)) {
			// eyepath は存在しない. lightpath が２つ以上.
			// => lightpath がカメラに直接ヒットしている可能性.

			if (camera->isPinhole()) {
				result = false;
			}
			else {
				const Vertex& eye = eyepath[0];
				const Vertex& endLight = lightpath[numLight - 1];

				auto dir = normalize(eye.pos - endLight.pos);
				ray r(endLight.pos + dir * AT_MATH_EPSILON, dir);

				hitrecord rec;
				bool isHit = scene->hit(r, AT_MATH_EPSILON, AT_MATH_INF, rec);

				vec3 posOnLens;
				vec3 posOnObjectplane;
				vec3 posOnImagesensor;

				const auto lens_t = camera->hitOnLens(
					r,
					posOnLens,
					posOnObjectplane,
					posOnImagesensor,
					px, py);

				if (AT_MATH_EPSILON < lens_t
					&& lens_t < rec.t)
				{
					// レイがレンズにヒット＆イメージセンサにヒット.
					px = aten::clamp(px, 0, m_width - 1);
					px = aten::clamp(px, 0, m_height - 1);
				}
				else {
					// lightサブパスを直接レンズにつなげようとしたが、遮蔽されたりイメージセンサにヒットしなかった場合、終わり.
					result = false;
				}
			}
		}
		else if ((numEye >= 2) && (numLight == 0)) {
			// direct hit to the light source

			// eyepath が２つ以上. lightpath がは存在しない.
			// => eyepath がライトに直接ヒットしている可能性.

			const Vertex& endEye = eyepath[numEye - 1];
			
			result = (endEye.mtrl && endEye.mtrl->isEmissive());
			firstRayDir = normalize(eyepath[1].pos - eyepath[0].pos);
			eyeOrg = eyepath[0].pos;
		}
		else {
			const Vertex& endEye = eyepath[numEye - 1];
			const Vertex& endLight = lightpath[numLight - 1];

			if ((numEye == 1) && (numLight >= 1)) {
				// light tracing

				// eyepath が１つだけ（カメラ上）.lightpath が１つ以上.
				// => カメラから直接ヒットしている可能性.

				const auto& eye = eyepath[0];
				auto dir = normalize(endLight.pos - eye.pos);

				ray r(eyepath[0].pos + dir * AT_MATH_EPSILON, dir);

				hitrecord rec;
				bool isHit = scene->hit(r, AT_MATH_EPSILON, AT_MATH_INF, rec);

				if (isHit) {
					if (rec.obj == endLight.obj) {
						result = true;
					}
				}

				firstRayDir = dir;
				eyeOrg = eyepath[0].pos;
			}
			else {
				// shadow ray connection

				if ((endEye.mtrl && endEye.mtrl->isSingular())
					|| (endLight.mtrl && endLight.mtrl->isSingular()))
				{
					// 端点がスペキュラやレンズだった場合は重みがゼロになりパス全体の寄与もゼロになるので、処理終わり.
					return false;
				}

				auto dir = normalize(endLight.pos - endEye.pos);
				ray r(endEye.pos + dir * AT_MATH_EPSILON, dir);

				hitrecord rec;
				bool isHit = scene->hit(r, AT_MATH_EPSILON, AT_MATH_INF, rec);

				if (isHit) {
					if (rec.obj == endLight.obj) {
						result = true;
					}
				}

				firstRayDir = normalize(eyepath[1].pos - eyepath[0].pos);
				eyeOrg = eyepath[0].pos;
			}
		}

		if (result && (px < 0 && py < 0))
		{
			camera->revertRayToPixelPos(
				ray(eyeOrg, firstRayDir),
				px, py);
		}

		return result && ((px >= 0) && (px < m_width) && (py >= 0) && (py < m_height));
	}

	vec3 BDPT::computeThroughput(
		const std::vector<Vertex>& vs,
		const CameraSampleResult& camsample,
		camera* camera)
	{
		vec3 f(1);

		const int num = (int)vs.size();

		for (int i = 0; i < num; i++) {
			const auto& vtx = vs[i];

			if (i == 0) {
				// カメラ.

				const auto& next = vs[i + 1];

				const auto W = camera->getWdash(
					next.pos,
					next.nml,
					camsample.posOnImageSensor,
					vtx.pos, //camsample.posOnLens,
					camsample.posOnObjectplane);

				f *= W;

				if (!camera->isPinhole()) {
					// Geometry term.
					vec3 wo = next.pos - vtx.pos;
					const auto dist2 = wo.squared_length();
					wo.normalize();

					const auto cos0 = dot(vtx.nml, wo);
					const auto cos1 = dot(next.nml, -wo);
					const auto G = aten::abs(cos0 * cos1 / dist2);

					f *= G;
				}
			}
			else if (i == (num - 1)) {
				// 光源.
				if (vtx.mtrl && vtx.mtrl->isEmissive()) {
					auto emit = vtx.mtrl->color();
					f *= emit;
				}
				else if (vtx.light) {
					auto emit = vtx.light->getLe();
					f *= emit;
				}
				else {
					f = vec3(0);
				}
			}
			else {
				const Vertex& fromVtx = vs[i - 1];
				const Vertex& toVtx = vs[i + 1];

				const vec3 wi = normalize(vtx.pos - fromVtx.pos);
				const vec3 wo = normalize(toVtx.pos - vtx.pos);
				
				vec3 brdf(0);
				
				if (vtx.mtrl) {
					brdf = vtx.mtrl->bsdf(vtx.nml, wi, wo, vtx.u, vtx.v);
				}
				else {
					continue;
				}

				if (vtx.mtrl->isSingular()) {
					// TODO
				}

				// Geometry term.
				const auto dist2 = (vtx.pos - fromVtx.pos).squared_length();
				const auto cos0 = dot(vtx.nml, wo);
				const auto cos1 = dot(toVtx.nml, -wo);
				const auto G = aten::abs(cos0 * cos1 / dist2);

				f = f * brdf * G;
			}

			auto l = f.squared_length();
			if (l == 0) {
				break;
			}
		}

		return f;
	}

	// compensated summation
	struct TKahanAdder {
		real sum, carry, y;

		TKahanAdder(const real b = 0.0)
		{
			sum = b;
			carry = 0.0;
			y = 0.0;
		}
		inline void add(const real b)
		{
			y = b - carry; 
			const real t = sum + y; 
			carry = (t - sum) - y; 
			sum = t;
		}
	};

	real BDPT::computePDF(
		const std::vector<Vertex>& vs,
		const CameraSampleResult& camsample,
		camera* camera,
		const int pathLength,
		const int specNumEye/*= -1*/,
		const int specNumLight/*= -1*/)
	{
		TKahanAdder sumPDFs(0.0);
		bool isSpecified = ((specNumEye >= 0) && (specNumLight >= 0));

		// number of eye subpath vertices
		for (int numEyeVertices = 0; numEyeVertices <= pathLength + 1; numEyeVertices++) {
			// number of light subpath vertices.
			int numLightVertices = (pathLength + 1) - numEyeVertices;

			// add all?
			if (isSpecified
				&& ((numEyeVertices != specNumEye) || (numLightVertices != specNumLight)))
			{
				continue;
			}

			real totalPdf = real(1);

			// sampling from the eye
			for (int i = -1; i <= numEyeVertices - 2; i++) {
				if (i < 0) {
					// Nothing is done...
				}
				else {
					const auto& vtx = vs[i];

					if (i == 0) {
						// カメラ上の１点をサンプルするPDF.

						const auto& next = vs[i + 1];

						const auto pdf = camera->getPdfImageSensorArea(
							next.pos,
							next.nml,
							camsample.posOnImageSensor,
							camsample.posOnLens,
							camsample.posOnObjectplane);
						
						totalPdf *= pdf;
					}
					else {
						// PDF of sampling ith vertex.
						const auto& fromVtx = vs[i - 1];
						const auto& toVtx = vs[i + 1];

						const vec3 wi = normalize(vtx.pos - fromVtx.pos);
						const vec3 wo = normalize(toVtx.pos - vtx.pos);

						real pdf = real(1);

						if (vtx.mtrl) {
							if (vtx.mtrl->isEmissive()) {
								// 完全拡散面とする.
								pdf = vtx.mtrl->pdf(vtx.nml, wi, wo, vtx.u, vtx.v);
							}
							else if (vtx.mtrl->isSingular()) {
								if (vtx.mtrl->isTranslucent()) {
									// TODO
								}
								else {
									pdf = real(1);
								}
							}
							else {
								pdf = vtx.mtrl->pdf(vtx.nml, wi, wo, vtx.u, vtx.v);
							}
						}
						else if (vtx.light) {
							// 完全拡散面とする.
							pdf = lambert::pdf(vtx.nml, wo);
						}

						const vec3 dv = toVtx.pos - vtx.pos;
						const real dist2 = dv.squared_length();
						const real c = dot(vtx.nml, dv);

						totalPdf *= pdf * aten::abs(c / dist2);
					}
				}
			}

			if (totalPdf != 0.0) {
				// sampling from the light source
				for (int i = -1; i <= numLightVertices - 2; i++) {
					if (i == -1) {
						// ライト上の１点をサンプリングするPDF.
						// PDF of sampling the light position (assume area-based sampling)
						const auto& lightSrc = vs[vs.size() - 1];
						auto sampleLightPdf = lightSrc.sampleLightPdf;
						totalPdf = totalPdf * sampleLightPdf;
					}
					else if (i == 0) {
						// ライトのマテリアルのPDF.
						// 完全拡散面とする.
						auto wo = normalize(vs[pathLength - 1].pos - vs[pathLength].pos);

						const auto& vtx = vs[pathLength];
						const auto& next = vs[pathLength - 1];

						auto pdf = lambert::pdf(vtx.nml, wo);

						vec3 dv = next.pos - vtx.pos;
						const real dist2 = dv.squared_length();
						dv.normalize();
						const real c = dot(vtx.nml, dv);
						
						totalPdf *= pdf * aten::abs(c / dist2);
					}
					else {
						// PDF of sampling (PathLength - i)th vertex
						auto wi = normalize(vs[pathLength - (i - 1)].pos - vs[pathLength - i].pos);
						auto wo = normalize(vs[pathLength - (i + 1)].pos - vs[pathLength - i].pos);

						const auto& vtx = vs[pathLength - i];
						const auto& next = vs[pathLength - (i + 1)];

						real pdf = real(1);

						if (vtx.mtrl) {
							if (vtx.mtrl->isEmissive()) {
								// 完全拡散面とする.
								pdf = vtx.mtrl->pdf(vtx.nml, wi, wo, vtx.u, vtx.v);
							}
							else if (vtx.mtrl->isSingular()) {
								if (vtx.mtrl->isTranslucent()) {
									// TODO
								}
								else {
									pdf = real(1);
								}
							}
							else {
								pdf = vtx.mtrl->pdf(vtx.nml, wi, wo, vtx.u, vtx.v);
							}

							const vec3 dv = next.pos - vtx.pos;
							const real dist2 = dv.squared_length();
							const real c = dot(vtx.nml, dv);

							totalPdf *= pdf * aten::abs(c / dist2);
						}
						else {
							// Hit on lens.

							// シーン上の点からレンズに入るレイ.
							ray r(next.pos, -wo);

							vec3 posOnLens;
							vec3 posOnObjectplane;
							vec3 posOnImagesensor;
							int x, y;

							auto lens_t = camera->hitOnLens(
								r,
								posOnLens,
								posOnObjectplane,
								posOnImagesensor,
								x, y);

							if (lens_t > AT_MATH_EPSILON) {
								// レイがレンズにヒット＆イメージセンサにヒット.

								// イメージセンサ上のサンプリング確率密度を計算.
								// イメージセンサの面積測度に関する確率密度をシーン上のサンプリング確率密度（面積測度に関する確率密度）に変換されている.
								const real imageSensorAreaPdf = camera->getPdfImageSensorArea(
									next.pos,
									next.nml,
									posOnImagesensor,
									posOnLens,
									posOnObjectplane);

								totalPdf *= imageSensorAreaPdf;
							}
						}
					}
				}
			}

			if (isSpecified
				&& (numEyeVertices == specNumEye) && (numLightVertices == specNumLight))
			{
				return totalPdf;
			}

			// sum the probability density (use Kahan summation algorithm to reduce numerical issues)
			sumPDFs.add(totalPdf);
		}

		return sumPDFs.sum;
	}

	// balance heuristic
	real BDPT::computeMISWeight(
		const std::vector<Vertex>& vs,
		const CameraSampleResult& camsample,
		camera* camera,
		const int numEyeVertices, 
		const int numLightVertices, 
		const int pathLength)
	{
		const real p_i = computePDF(vs, camsample, camera, pathLength, numEyeVertices, numLightVertices);
		const real p_all = computePDF(vs, camsample, camera, pathLength);

		if ((p_i == 0.0) || (p_all == 0.0)) {
			return 0.0;
		}

		auto mis = std::max(std::min(p_i / p_all, real(1)), real(0));
		return mis;
	}

	void BDPT::combinePath(
		std::vector<Result>& result,
		const std::vector<Vertex>& eyepath,
		const std::vector<Vertex>& lightpath,
		scene* scene,
		const CameraSampleResult& camsample,
		camera* camera)
	{
		// NOTE
		// MaxEvents = the maximum number of vertices

		static const int MinPathLength = 3;		// avoid sampling direct illumination
		static const int MaxPathLength = 20;

		const auto& beginEye = eyepath[0];

		for (int pathLength = MinPathLength; pathLength <= MaxPathLength; pathLength++) {
			for (int numEyeVertices = 0; numEyeVertices <= pathLength + 1; numEyeVertices++) {
				const int numLightVertices = (pathLength + 1) - numEyeVertices;

				if (numEyeVertices > eyepath.size()) {
					continue;
				}
				if (numLightVertices > lightpath.size()) {
					continue;
				}

				int x = beginEye.x;
				int y = beginEye.y;

				// check the path visibility
				if (!isConnectable(scene, camera, eyepath, numEyeVertices, lightpath, numLightVertices, x, y)) {
					continue;
				}

				// construct a full path
				std::vector<Vertex> sampledPath(numEyeVertices + numLightVertices);
				for (int i = 0; i < numEyeVertices; i++) {
					sampledPath[i] = eyepath[i];
				}
				for (int i = 0; i < numLightVertices; i++) {
					// lightpath は 0 がライトの位置で、ライトから始まるので、subpaths には最後が終端（ライト）になるように格納する.
					sampledPath[pathLength - i] = lightpath[i];
				}
				auto numSampledPath = numEyeVertices + numLightVertices;

				// evaluate the path
				vec3 f = computeThroughput(sampledPath, camsample, camera);
				real p = computePDF(sampledPath, camsample, camera, pathLength, numEyeVertices, numLightVertices);
				real w = computeMISWeight(sampledPath, camsample, camera, numEyeVertices, numLightVertices, pathLength);

				if ((w <= 0.0) || (p <= 0.0)) {
					continue;
				}

				vec3 c = f / (w * p);
				auto _max = std::max(c.x, std::max(c.y, c.z));
				auto _min = std::min(c.x, std::min(c.y, c.z));
				if (_max < 0.0 || _min < 0.0) {
					continue;
				}

				// store the pixel contribution
				result.push_back(Result(
					c, 
					x, y,
					numEyeVertices < 1 ? false : true));
			}
		}
	}

	void BDPT::render(
		Destination& dst,
		scene* scene,
		camera* camera)
	{
		m_width = dst.width;
		m_height = dst.height;
		uint32_t samples = dst.sample;

		m_maxDepth = dst.maxDepth;

		// TODO
		/*
		m_rrDepth = dst.russianRouletteDepth;

		if (m_rrDepth > m_maxDepth) {
			m_rrDepth = m_maxDepth - 1;
		}
		*/

		auto threadnum = thread::getThreadNum();

		std::vector<std::vector<vec4>> image(threadnum);

		for (int i = 0; i < threadnum; i++) {
			image[i].resize(m_width * m_height);
		}

#if defined(ENABLE_OMP) && !defined(BDPT_DEBUG)
#pragma omp parallel
#endif
		{
			auto idx = thread::getThreadIdx();

			auto time = timer::getSystemTime();

#if defined(ENABLE_OMP) && !defined(BDPT_DEBUG)
#pragma omp for
#endif
			for (int y = 0; y < m_height; y++) {
				for (int x = 0; x < m_width; x++) {

					for (uint32_t i = 0; i < samples; i++) {
						//XorShift rnd((y * height * 4 + x * 4) * samples + i + 1);
						//Halton rnd((y * height * 4 + x * 4) * samples + i + 1);
						//Sobol rnd((y * height * 4 + x * 4) * samples + i + 1 + time.milliSeconds);
						Sobol rnd((y * m_height * 4 + x * 4) * samples + i + 1);
						UniformDistributionSampler sampler(&rnd);

						std::vector<Vertex> eyevs;
						std::vector<Vertex> lightvs;

						auto eyeRes = genEyePath(eyevs, x, y, &sampler, scene, camera);

						if (eyeRes.x >= 0 && eyeRes.y >= 0) {
							// Hit ray light source directly.
							auto pos = eyeRes.y * m_width + eyeRes.x;
							image[idx][pos] += vec4(eyeRes.contrib, 1);
						}

						auto lightNum = scene->lightNum();
						for (uint32_t i = 0; i < lightNum; i++) {
							auto light = scene->getLight(i);
							auto lightRes = genLightPath(lightvs, light, &sampler, scene, camera);

							if (lightRes.x >= 0 && lightRes.y >= 0) {
								// Hit ray camera lens directly.
								auto pos = lightRes.y * m_width + lightRes.x;
								image[idx][pos] += vec4(lightRes.contrib / (real)(m_width * m_height), 1);
							}

							std::vector<Result> result;

							combinePath(result, eyevs, lightvs, scene, eyeRes.camsample, camera);

							for (uint32_t n= 0; n < (uint32_t)result.size(); n++) {
								const auto& res = result[n];

								auto pos = res.y * m_width + res.x;
								image[idx][pos] += vec4(res.contrib, 1);
							}
						}
					}
				}
			}
		}

		std::vector<vec4> tmp(m_width * m_height);

		for (int i = 0; i < threadnum; i++) {
			auto& img = image[i];
			for (int y = 0; y < m_height; y++) {
				for (int x = 0; x < m_width; x++) {
					int pos = y * m_width + x;

					auto clr = img[pos] / samples;
					clr.w = 1;

					tmp[pos] += clr;
				}
			}
		}

#if defined(ENABLE_OMP) && !defined(BDPT_DEBUG)
#pragma omp parallel for
#endif
		for (int y = 0; y < m_height; y++) {
			for (int x = 0; x < m_width; x++) {
				int pos = y * m_width + x;

				auto clr = tmp[pos];
				clr.w = 1;

				dst.buffer->put(x, y, clr);
			}
		}
	}
}