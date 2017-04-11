#include "renderer/bdpt2.h"
#include "misc/thread.h"
#include "misc/timer.h"
#include "material/lambert.h"
#include "sampler/xorshift.h"
#include "sampler/halton.h"
#include "sampler/sobolproxy.h"
#include "sampler/UniformDistributionSampler.h"

//#define BDPT_DEBUG

#ifdef BDPT_DEBUG
#pragma optimize( "", off) 
#endif

namespace aten
{
	static inline real russianRoulette(const vec3& v)
	{
		auto t = normalize(v);
		auto p = std::max(t.r, std::max(t.g, t.b));
		return p;
	}

	BDPT2::Result BDPT2::genEyePath(
		std::vector<Vertex>& vs,
		int x, int y,
		sampler* sampler,
		scene* scene,
		camera* camera) const
	{
		real u = (real)(x + sampler->nextSample()) / (real)m_width;
		real v = (real)(y + sampler->nextSample()) / (real)m_height;

		auto camsample = camera->sample(u, v, sampler);

		// レンズ上の頂点（x0）を頂点リストに追加.
		vs.push_back(Vertex(
			camsample.posOnLens,
			camsample.nmlOnLens,
			camsample.nmlOnLens,
			ObjectType::Lens,
			camsample.pdfOnLens,
			vec3(1),
			vec3(0),
			nullptr,
			nullptr,
			0, 0));

		int depth = 0;

		ray ray = camsample.r;

		vec3 throughput(1);
		real totalAreaPdf = real(1);

		vec3 prevNormal = camera->getDir();
		real sampledPdf = real(1);

		while (depth < m_maxDepth) {
			hitrecord rec;
			if (!scene->hit(ray, AT_MATH_EPSILON, AT_MATH_INF, rec)) {
				break;
			}

			// 交差位置の法線.
			// 物体からのレイの入出を考慮.
			vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

			// ロシアンルーレットによって、新しい頂点を「実際に」サンプリングし、生成するのかどうかを決定する.
			auto rrProb = russianRoulette(throughput);
			auto rr = sampler->nextSample();
			if (rr >= rrProb) {
				break;
			}

			// 新しい頂点がサンプリングされたので、トータルの確率密度に乗算する.
			totalAreaPdf *= rrProb;

			const vec3 toNextVtx = ray.org - rec.p;

			if (depth == 0) {
				// x1のサンプリング確率密度はイメージセンサ上のサンプリング確率密度を変換することで求める,
				auto pdfOnImageSensor = camera->getPdfImageSensorArea(
					rec.p,
					orienting_normal,
					camsample.posOnImageSensor,
					camsample.posOnLens,
					camsample.posOnObjectplane);

				totalAreaPdf *= pdfOnImageSensor;

				// 幾何的な係数計算 + センサーセンシティビティの項を計算する
				// x1 -> x0への放射輝度が最終的にイメージセンサに与える寄与度
				auto W_dash = camera->getWdash(
					rec.p,
					orienting_normal,
					camsample.posOnImageSensor,
					camsample.posOnLens,
					camsample.posOnObjectplane);

				throughput = W_dash * throughput;
			}
			else {
				// 新しい頂点をサンプリングするための確率密度関数は立体角測度に関するものであったため、これを面積速度に関する確率密度関数に変換する.
				const real c = dot(normalize(toNextVtx), orienting_normal);
				const real dist2 = toNextVtx.squared_length();
				const real areaPdf = sampledPdf * (c / dist2);

				totalAreaPdf *= areaPdf;
			}

			// ジオメトリターム.
			const real c0 = dot(normalize(toNextVtx), orienting_normal);
			const real c1 = dot(normalize(-toNextVtx), prevNormal);
			const real dist2 = toNextVtx.squared_length();
			const double G = c0 * c1 / dist2;
			throughput = G * throughput;

			// 光源にヒットしたらそこで追跡終了.
			if (rec.mtrl->isEmissive()) {
				vec3 bsdf = lambert::bsdf(rec.mtrl, rec.u, rec.v);

				vs.push_back(Vertex(
					rec.p,
					rec.normal,
					orienting_normal,
					ObjectType::Light,
					totalAreaPdf,
					throughput,
					bsdf,
					rec.obj,
					rec.mtrl,
					rec.u, rec.v));

				vec3 emit = rec.mtrl->color();
				vec3 contrib = throughput * emit;

				return std::move(Result(contrib, x, y, true));
			}

			auto sampling = rec.mtrl->sample(ray, orienting_normal, rec, sampler, rec.u, rec.v);

			// 新しい頂点を頂点リストに追加する.
			vs.push_back(Vertex(
				rec.p,
				rec.normal,
				orienting_normal,
				ObjectType::Object,
				totalAreaPdf,
				throughput,
				sampling.bsdf,
				rec.obj,
				rec.mtrl,
				rec.u, rec.v));

			sampledPdf = sampling.pdf;
			throughput *= sampling.bsdf;

			vec3 nextDir = normalize(sampling.dir);
			
			if (rec.mtrl->isSingular()) {
				// For canceling cosine term.
				auto costerm = dot(normalize(toNextVtx), orienting_normal);
				throughput /= costerm;

				// Just only for refraction.
				// Cancel probability to select reflection or refraction.
				throughput *= sampling.subpdf;
			}

			// refractionの反射、屈折の確率を掛け合わせる.
			// refraction以外では 1 なので影響はない.
			totalAreaPdf *= sampling.subpdf;

			ray = aten::ray(rec.p + nextDir * AT_MATH_EPSILON, nextDir);

			prevNormal = orienting_normal;
			depth++;
		}

		return std::move(Result(vec3(), -1, -1, false));
	}

	BDPT2::Result BDPT2::genLightPath(
		std::vector<Vertex>& vs,
		aten::Light* light,
		sampler* sampler,
		scene* scene,
		camera* camera) const
	{
		// TODO
		// Only AreaLight...

		// 光源上にサンプル点生成（y0）.
		auto res = light->getSamplePosNormalPdf(sampler);
		auto posOnLight = std::get<0>(res);
		auto nmlOnLight = std::get<1>(res);
		auto pdfOnLight = std::get<2>(res);

		// 確率密度の積を保持（面積測度に関する確率密度）.
		double totalAreaPdf = pdfOnLight;

		// 光源上に生成された頂点を頂点リストに追加.
		vs.push_back(Vertex(
			posOnLight,
			nmlOnLight,
			nmlOnLight,
			ObjectType::Light,
			totalAreaPdf,
			vec3(0),
			light->getLe(),
			light));

		// 現在の放射輝度（モンテカルロ積分のスループット）.
		// 本当は次の頂点（y1）が決まらないと光源からその方向への放射輝度値は決まらないが、今回は完全拡散光源を仮定しているので、方向に依らずに一定の値になる.
		vec3 throughput = light->getLe();

		int depth = 0;

		// 完全拡散光源を仮定しているので、Diffuse面におけるサンプリング方法と同じものをつかって次の方向を決める.
		nmlOnLight = normalize(nmlOnLight);
		vec3 dir = lambert::sampleDirection(nmlOnLight, sampler);
		real sampledPdf = lambert::pdf(nmlOnLight, dir);
		vec3 prevNormal = nmlOnLight;

		ray ray = aten::ray(posOnLight + dir * AT_MATH_EPSILON, dir);

		while (depth < m_maxDepth) {
			hitrecord rec;
			bool isHit = scene->hit(ray, AT_MATH_EPSILON, AT_MATH_INF, rec);

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
				// TODO
			}

			if (!isHit) {
				break;
			}

			// 交差位置の法線.
			// 物体からのレイの入出を考慮.
			vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

			// ロシアンルーレットによって、新しい頂点を「実際に」サンプリングし、生成するのかどうかを決定する.
			auto rrProb = russianRoulette(throughput);
			auto rr = sampler->nextSample();
			if (rr >= rrProb) {
				break;
			}

			// 新しい頂点がサンプリングされたので、トータルの確率密度に乗算する.
			totalAreaPdf *= rrProb;

			const vec3 toNextVtx = ray.org - rec.p;

			{
				// 新しい頂点をサンプリングするための確率密度関数は立体角測度に関するものであったため、これを面積測度に関する確率密度関数に変換する.
				const real c = dot(normalize(toNextVtx), orienting_normal);
				const real dist2 = toNextVtx.squared_length();
				const real areaPdf = sampledPdf * (c / dist2);

				// 全ての頂点をサンプリングする確率密度の総計を出す。
				totalAreaPdf *= areaPdf;
			}

			{
				// ジオメトリターム.
				const real c0 = dot(normalize(toNextVtx), orienting_normal);
				const real c1 = dot(normalize(-toNextVtx), prevNormal);
				const real dist2 = toNextVtx.squared_length();
				const real G = c0 * c1 / dist2;

				throughput = G * throughput;
			}

			auto sampling = rec.mtrl->sample(ray, orienting_normal, rec, sampler, rec.u, rec.v);

			// 新しい頂点を頂点リストに追加する.
			vs.push_back(Vertex(
				rec.p,
				rec.normal,
				orienting_normal,
				ObjectType::Object,
				totalAreaPdf,
				throughput,
				sampling.bsdf,
				rec.obj,
				rec.mtrl,
				rec.u, rec.v));

			sampledPdf = sampling.pdf;
			throughput *= sampling.bsdf;

			vec3 nextDir = normalize(sampling.dir);

			if (rec.mtrl->isSingular()) {
				// For canceling cosine term.
				auto costerm = dot(normalize(toNextVtx), orienting_normal);
				throughput /= costerm;

				// Just only for refraction.
				// Cancel probability to select reflection or refraction.
				throughput *= sampling.subpdf;
			}

			// refractionの反射、屈折の確率を掛け合わせる.
			// refraction以外では 1 なので影響はない.
			totalAreaPdf *= sampling.subpdf;

			ray = aten::ray(rec.p + nextDir * AT_MATH_EPSILON, nextDir);

			prevNormal = orienting_normal;
			depth++;
		}

		return std::move(Result(vec3(), -1, -1, false));
	}

	// 頂点fromから頂点nextをサンプリングしたとするとき、面積測度に関するサンプリング確率密度を計算する.
	real BDPT2::computAreaPdf(
		camera* camera,
		const std::vector<const Vertex*>& vs,
		const int prev_idx,			// 頂点curの前の頂点のインデックス.
		const int cur_idx,          // 頂点curのインデックス.
		const int next_idx) const   // 頂点nextのインデックス.
	{
		const Vertex& curVtx = *vs[cur_idx];
		const Vertex& nextVtx = *vs[next_idx];

		Vertex const* prevVtx = nullptr;
		if (0 <= prev_idx && prev_idx < vs.size()) {
			prevVtx = vs[prev_idx];
		}

		const vec3 to = nextVtx.pos - curVtx.pos;
		const vec3 normalizedTo = normalize(to);

		real pdf = real(0);

		// 頂点fromのオブジェクトの種類によって、その次の頂点のサンプリング確率密度の計算方法が変わる.
		if (curVtx.objType == ObjectType::Light) {
			// TODO
			// Lightは完全拡散面として扱う.
			pdf = lambert::pdf(curVtx.orienting_normal, normalizedTo);
		}
		else if (curVtx.objType == ObjectType::Lens) {
			// レンズ上の点からシーン上の点をサンプリングするときの面積測度に関する確率密度を計算.
			// イメージセンサ上の点のサンプリング確率密度を元に変換する.

			// シーン上の点からレンズに入るレイ.
			ray r(nextVtx.pos, -normalizedTo);

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
					nextVtx.pos,
					nextVtx.orienting_normal,
					posOnImagesensor,
					posOnLens,
					posOnObjectplane);

				return imageSensorAreaPdf;
			}
			else {
				return real(0);
			}
		}
		else {
			if (prevVtx) {
				const vec3 wi = normalize(curVtx.pos - prevVtx->pos);
				const vec3 wo = normalize(nextVtx.pos - curVtx.pos);

				if (curVtx.mtrl->isTranslucent()) {
					// TODO
				}
				else {
					pdf = curVtx.mtrl->pdf(curVtx.orienting_normal, wi, wo, curVtx.u, curVtx.v);
				}
			}
		}

		// 次の頂点の法線を、現在の頂点からの方向ベクトルに基づいて改めて求める.
		const vec3 next_new_orienting_normal = dot(to, nextVtx.nml) < 0.0
			? nextVtx.nml
			: -nextVtx.nml;

		// 立体角測度に関する確率密度を面積測度に関する確率密度に変換.
		const real c = dot(-normalizedTo, next_new_orienting_normal);
		const real dist2 = to.squared_length();;
		pdf *= c / dist2;

		return pdf;
	}

	real BDPT2::computeMISWeight(
		camera* camera,
		real totalAreaPdf,
		const std::vector<Vertex>& eye_vs,
		int numEyeVtx,
		const std::vector<Vertex>& light_vs,
		int numLightVtx) const
	{
		// NOTE
		// https://www.slideshare.net/h013/edubpt-v100
		// p157 - p167

		// 光源上のサンプリング確率.
		const auto& beginLight = light_vs[0];
		const real areaPdf_y0 = beginLight.totalAreaPdf;

		// カメラ上のサンプリング確率.
		const auto& beginEye = eye_vs[0];
		const real areaPdf_x0 = beginEye.totalAreaPdf;

		std::vector<const Vertex*> vs(numEyeVtx + numLightVtx);

		// 頂点を一列に並べる。
		// vs[0] = y0, vs[1] = y1, ... vs[k-1] = x1, vs[k] = x0
		
		// lightサブパス.
		for (int i = 0; i < numLightVtx; ++i) {
			vs[i] = &light_vs[i];
		}

		// eyeサブパス.
		for (int i = numEyeVtx - 1; i >= 0; --i) {
			vs[numLightVtx + numEyeVtx - 1 - i] = &eye_vs[i];
		}

		// 終点のインデックス.
		const int k = numLightVtx + numEyeVtx - 1;

		// pi1/pi を計算.
		std::vector<real> pi1_pi(numLightVtx + numEyeVtx);
		{
			{
				const auto* vtx = vs[0];
				auto rr = russianRoulette(vtx->throughput);
				auto areaPdf_y1 = computAreaPdf(camera, vs, 2, 1, 0);
				pi1_pi[0] = areaPdf_y0 / (areaPdf_y1 * rr);
			}

			// ロシアンルーレットの確率は打ち消しあうのでいらない？
			for (int i = 1; i < k; i++)
			{
				auto a = computAreaPdf(camera, vs, i - 2, i - 1, i);
				auto b = computAreaPdf(camera, vs, i + 2, i + 1, i);
				pi1_pi[i] = a / b;
			}

			{
				const auto* vtx = vs[k];
				auto rr = russianRoulette(vtx->throughput);
				auto areaPdf_x1 = computAreaPdf(camera, vs, k - 2, k - 1, k);
				pi1_pi[k] = (areaPdf_x1 * rr) / areaPdf_x0;
			}
		}

		// pを求める
		std::vector<real> p(numEyeVtx + numLightVtx + 1);
		{
			// 真ん中にtotalAreaPdfをセット.
			p[numLightVtx] = totalAreaPdf;

			// 真ん中を起点に半分ずつ計算.

			// lightサブパス.
			for (int i = numLightVtx; i <= k; ++i) {
				p[i + 1] = p[i] * pi1_pi[i];
			}

			// eyeサブパス.
			for (int i = numLightVtx - 1; i >= 0; --i) {
				p[i] = p[i + 1] / pi1_pi[i];
			}

			for (int i = 0; i < vs.size(); ++i) {
				const auto& vtx = *vs[i];

				// 方向が一意に決まるので、影響を及ぼさない.
				if (vtx.mtrl && vtx.mtrl->isSingular()) {
					p[i] = 0.0;
					p[i + 1] = 0.0;
				}
			}
		}

		// Power-heuristic
		real misWeight = real(0);
		for (int i = 0; i < p.size(); ++i) {
			const real v = p[i] / p[numLightVtx];
			misWeight += v * v; // beta = 2
		}

		if (misWeight > real(0)) {
			misWeight = real(1) / misWeight;
		}

		return misWeight;
	}

	void BDPT2::combine(
		const int x, const int y,
		std::vector<Result>& result,
		const std::vector<Vertex>& eye_vs,
		const std::vector<Vertex>& light_vs,
		scene* scene,
		camera* camera) const
	{
		const int eyeNum = (int)eye_vs.size();
		const int lightNum = (int)light_vs.size();

		for (int numEyeVtx = 1; numEyeVtx <= eyeNum; ++numEyeVtx)
		{
			for (int numLightVtx = 1; numLightVtx <= lightNum; ++numLightVtx)
			{
				int targetX = x;
				int targetY = y;

				// それぞれのパスの端点.
				const Vertex& eye_end = eye_vs[numEyeVtx - 1];
				const Vertex& light_end = light_vs[numLightVtx - 1];

				// トータルの確率密度計算.
				const real totalAreaPdf = eye_end.totalAreaPdf * light_end.totalAreaPdf;
				if (totalAreaPdf == 0) {
					// 何もなかったので、何もしない.
					continue;
				}

				// MCスループット.
				vec3 eyeThroughput = eye_end.throughput;
				vec3 lightThroughput = light_end.throughput;

				// 頂点を接続することで新しく導入される項.
				vec3 throughput(1);

				// numLightVtx == 1のとき、非完全拡散光源の場合は相手の頂点の位置次第でMCスループットが変化するため改めて光源からの放射輝度値を計算する.
				if (numLightVtx == 1) {
					// TODO
					// 今回は完全拡散光源なので単純にemissionの値を入れる.
					const auto& lightVtx = light_vs[0];
					lightThroughput = lightVtx.light->getLe();
				}

				// 端点間が接続できるか.
				const vec3 lightEndToEyeEnd = eye_end.pos - light_end.pos;
				ray r(light_end.pos, normalize(lightEndToEyeEnd));

				hitrecord rec;
				bool isHit = scene->hit(r, AT_MATH_EPSILON, AT_MATH_INF, rec);

				if (eye_end.objType == ObjectType::Lens) {
					// lightサブパスを直接レンズにつなげる.
					vec3 posOnLens;
					vec3 posOnObjectplane;
					vec3 posOnImagesensor;
					int px, py;

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
						targetX = aten::clamp(px, 0, m_width - 1);
						targetY = aten::clamp(px, 0, m_height - 1);

						const real W_dash = camera->getWdash(
							rec.p,
							rec.normal,
							posOnImagesensor,
							posOnLens,
							posOnObjectplane);

						throughput *= W_dash;
					}
					else {
						// lightサブパスを直接レンズにつなげようとしたが、遮蔽されたりイメージセンサにヒットしなかった場合、終わり.
						continue;
					}
				}
				else if (eye_end.objType == ObjectType::Light) {
					// eyeサブパスの端点が光源（反射率0）だった場合は重みがゼロになりパス全体の寄与もゼロになるので、処理終わり.
					// 光源は反射率0を想定している.
					continue;
				}
				else {
					if (eye_end.mtrl->isSingular()) {
						// eyeサブパスの端点がスペキュラやだった場合は重みがゼロになりパス全体の寄与もゼロになるので、処理終わり.
						// スペキュラの場合は反射方向で一意に決まり、lighサブパスの端点への方向が一致する確率がゼロ.
						continue;
					}
					else {
						// 端点同士が別の物体で遮蔽されるかどうかを判定する。遮蔽されていたら処理終わり.
						const real len = (eye_end.pos - rec.p).length();
						if (len >= AT_MATH_EPSILON) {
							continue;
						}

						const auto& bsdf = eye_end.bsdf;
						throughput *= bsdf;
					}
				}

				if (light_end.objType == ObjectType::Lens
					|| light_end.objType == ObjectType::Light)
				{
					// lightサブパスの端点がレンズだった場合は重みがゼロになりパス全体の寄与もゼロになるので、処理終わり.
					// レンズ上はスペキュラとみなす.

					// eyeサブパスの端点が光源（反射率0）だった場合は重みがゼロになりパス全体の寄与もゼロになるので、処理終わり.
					// 光源は反射率0を想定している.
				}
				else {
					if (light_end.mtrl->isSingular()) {
						// eyeサブパスの端点がスペキュラやだった場合は重みがゼロになりパス全体の寄与もゼロになるので、処理終わり.
						// スペキュラの場合は反射方向で一意に決まり、lighサブパスの端点への方向が一致する確率がゼロ.
						continue;
					}
					else {
						const auto& bsdf = light_end.bsdf;
						throughput *= bsdf;
					}
				}

				// 端点間のジオメトリファクタ
				{
					real cx = dot(normalize(-lightEndToEyeEnd), eye_end.orienting_normal);
					cx = std::max(cx, real(0));

					real cy = dot(normalize(lightEndToEyeEnd), light_end.orienting_normal);
					cy = std::max(cy, real(0));

					const real dist2 = lightEndToEyeEnd.squared_length();

					const real G = cx * cy / dist2;

					throughput *= G;
				}

				// MIS.
				const real misWeight = computeMISWeight(
					camera,
					totalAreaPdf,
					eye_vs, numEyeVtx,
					light_vs, numLightVtx);

				if (misWeight <= real(0)) {
					continue;
				}

				// 最終的なモンテカルロコントリビューション =
				//	MIS重み.
				//	* 頂点を接続することで新しく導入される項.
				//  * eyeサブパススループット.
				//	* lightサブパススループット.
				//	/ パスのサンプリング確率密度の総計.
				const vec3 contrib = misWeight * throughput * eyeThroughput * lightThroughput / totalAreaPdf;
				result.push_back(Result(
					contrib,
					targetX, targetY,
					false));
			}
		}
	}
}