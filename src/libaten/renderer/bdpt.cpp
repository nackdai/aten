#include "renderer/bdpt.h"
#include "misc/thread.h"
#include "misc/timer.h"

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
		scene* scene) const
	{
		uint32_t depth = 0;

		ray ray = r;

		vec3 throughput(1);
		vec3 contrib(0);

		while (depth < maxDepth) {
			hitrecord rec;
			if (!scene->hit(ray, AT_MATH_EPSILON, AT_MATH_INF, rec)) {
				return vec3(0);
			}

			vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

			vs.push_back(Vertex(
				rec.p,
				orienting_normal,
				rec.obj,
				rec.mtrl,
				rec.u, rec.v));

			if (isEyePath && rec.mtrl->isEmissive()) {
				auto emit = rec.mtrl->color();
				contrib = throughput * emit;
				break;
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

	vec3 BDPT::genEyePath(
		std::vector<Vertex>& vs,
		real u, real v,
		sampler* sampler,
		scene* scene,
		camera* cam) const
	{
		auto camsample = cam->sample(u, v, sampler);

		// カメラ.
		vs.push_back(Vertex(
			camsample.posOnLens,
			camsample.nmlOnLens,
			nullptr,
			nullptr,
			0, 0));

		auto contrib = trace(
			true,
			vs,
			m_maxDepth,
			camsample.r,
			sampler,
			scene);

		return std::move(contrib);
	}

	void BDPT::genLightPath(
		std::vector<Vertex>& vs,
		Light* light,
		sampler* sampler,
		scene* scene) const
	{
		// TODO
		// Only AreaLight...

		auto res = light->getSamplePosAndNormal(sampler);
		auto pos = std::get<0>(res);
		auto nml = std::get<1>(res);

		// 光源.
		vs.push_back(Vertex(
			pos,
			nml,
			light,
			nullptr,
			0, 0));

		vec3 dir;
		{
			// normalの方向を基準とした正規直交基底(w, u, v)を作る.
			// この基底に対する半球内で次のレイを飛ばす.
			vec3 n, t, b;

			n = nml;

			// nと平行にならないようにする.
			if (fabs(n.x) > 0.1) {
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

		trace(
			false,
			vs,
			m_maxDepth,
			r,
			sampler,
			scene);
	}

	bool BDPT::isConnectable(
		scene* scene,
		camera* cam,
		const std::vector<Vertex>& eyepath,
		const int numEye,
		const std::vector<Vertex>& lightpath,
		const int numLight) const
	{
		vec3 firstRayDir; // Direction from camera origin.
		vec3 eyeOrg;

		bool result;

		if ((numEye == 0) && (numLight >= 2)) {
			// TODO
			// no direct hit to the film (pinhole)

			// eyepath は存在しない. lightpath が２つ以上.
			// => lightpath がカメラに直接ヒットしている可能性.
			
			result = false;
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

				AT_ASSERT(endEye.mtrl);
				AT_ASSERT(endLight.mtrl);

				if (endEye.mtrl->isSingular()
					|| endLight.mtrl->isSingular())
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

		int x = -1;
		int y = -1;

		if (result) {
			cam->revertRayToPixelPos(
				ray(eyeOrg, firstRayDir),
				x, y);
		}

		return result && ((x >= 0) && (x < m_width) && (y >= 0) && (y < m_height));
	}

	vec3 computeThroughput(
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
					camsample.posOnLens,
					camsample.posOnObjectplane);

				f *= W;
			}
			else if (i == (num - 1)) {
				// 光源.
				if (vtx.mtrl && vtx.mtrl->isEmissive()) {
					auto emit = vtx.mtrl->color();
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

				AT_ASSERT(vtx.mtrl);
				
				if (vtx.mtrl) {
					brdf = vtx.mtrl->bsdf(vtx.nml, wi, wo, vtx.u, vtx.v);
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

	real computePDF(
		const std::vector<Vertex>& vs,
		const CameraSampleResult& camsample,
		camera* camera,
		const int pathLenght,
		const int specNumEye = -1,
		const int specNumLight = -1)
	{
		TKahanAdder sumPDFs(0.0);
		bool isSpecified = ((specNumEye >= 0) && (specNumLight >= 0));

		// number of eye subpath vertices
		for (int numEyeVertices = 0; numEyeVertices <= pathLenght + 1; numEyeVertices++) {
			// number of light subpath vertices.
			int numLightVertices = (pathLenght + 1) - numEyeVertices;

			// TODO
			// Only pinhole...
			if (numEyeVertices == 0) {
				continue;
			}

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
						// カメラ.

						const auto& next = vs[i + 1];

						const auto pdf = camera->getPdfImageSensorArea(
							next.pos,
							next.nml,
							camsample.posOnImageSensor,
							camsample.posOnLens,
							camsample.posOnObjectplane);
						
						const vec3 dv = next.pos - vtx.pos;
						const real dist2 = dv.squared_length();
						const real c = dot(vtx.nml, dv);

						totalPdf *= pdf * aten::abs(c / dist2);
					}
					else {
						// PDF of sampling ith vertex.
						const auto& fromVtx = vs[i - 1];
						const auto& toVtx = vs[i + 1];

						const vec3 wi = normalize(vtx.pos - fromVtx.pos);
						const vec3 wo = normalize(toVtx.pos - vtx.pos);

						real pdf = real(1);

						if (vtx.mtrl->isEmissive()) {
							
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

						const vec3 dv = toVtx.pos - vtx.pos;
						const real dist2 = dv.squared_length();
						const real c = dot(vtx.nml, dv);

						totalPdf *= pdf * aten::abs(c / dist2);
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
	real computeMISWeight(
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
		else {
			auto mis = std::max(std::min(p_i / p_all, real(1)), real(0));
		}
	}

	void BDPT::contribution(
		std::vector<vec3>& result,
		const std::vector<Vertex>& eyepath,
		const std::vector<Vertex>& lightpath,
		scene* scene,
		const CameraSampleResult& camsample,
		camera* camera) const
	{
		// NOTE
		// MaxEvents = the maximum number of vertices

		static const int MinPathLength = 3;		// avoid sampling direct illumination
		static const int MaxPathLength = 20;

		for (int pathLength = MinPathLength; pathLength <= MaxPathLength; pathLength++) {
			for (int numEyeVertices = 0; numEyeVertices <= pathLength + 1; numEyeVertices++) {
				const int numLightVertices = (pathLength + 1) - numEyeVertices;

				if (numEyeVertices == 0) {
					// TODO
					// no direct hit to the film (pinhole)
					continue;
				}
				if (numEyeVertices > eyepath.size()) {
					continue;
				}
				if (numLightVertices > eyepath.size()) {
					continue;
				}

				// check the path visibility
				if (!isConnectable(scene, camera, eyepath, numEyeVertices, lightpath, numLightVertices)) {
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
				auto m = std::max(c.x, std::max(c.y, c.z));
				if (m < 0.0) {
					continue;
				}

				// store the pixel contribution
				result.push_back(c);
			}
		}
	}
}