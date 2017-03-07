#include "renderer/erpt.h"
#include "misc/thread.h"
#include "sampler/xorshift.h"
#include "sampler/halton.h"
#include "sampler/sobolproxy.h"
#include "misc/color.h"

namespace aten
{
	static const real MutateDistance = real(0.05);

	class ERPTSampler : public sampler {
	public:
		ERPTSampler(random* rnd);
		virtual ~ERPTSampler() {}

	public:
		virtual real nextSample() override final;

		virtual random* getRandom() override final
		{
			return m_rnd;
		}

		void reset()
		{
			m_usedRandCoords = 0;
		}

		void mutate();

	private:
		inline real mutate(real value);

	private:
		random* m_rnd{ nullptr };

		int m_usedRandCoords{ 0 };

		std::vector<real> m_primarySamples;
	};

	ERPTSampler::ERPTSampler(random* rnd)
		: m_rnd(rnd)
	{
		static const int initsize = 32;
		m_primarySamples.resize(initsize);

		for (int i = 0; i < m_primarySamples.size(); i++) {
			m_primarySamples[i] = rnd->next01();
		}
	}

	real ERPTSampler::mutate(real value)
	{
		// Same as LuxRender?

		auto r = m_rnd->next01();

		double v = MutateDistance * (2.0 * r - 1.0);
		value += v;

		if (value > 1.0) {
			value -= 1.0;
		}

		if (value < 0.0) {
			value += 1.0;
		}

		return value;
	}

	real ERPTSampler::nextSample()
	{
		if (m_primarySamples.size() <= m_usedRandCoords) {
			const int now_max = m_primarySamples.size();

			// 拡張する.
			m_primarySamples.resize(m_primarySamples.size() * 1.5);

			// 拡張した部分に値を入れる.
			for (int i = now_max; i < m_primarySamples.size(); i++) {
				m_primarySamples[i] = m_rnd->next01();
			}
		}

		m_usedRandCoords++;

		auto ret = m_primarySamples[m_usedRandCoords - 1];

		return ret;
	}

	void ERPTSampler::mutate()
	{
		for (int i = 0; i < m_primarySamples.size(); i++) {
			auto prev = m_primarySamples[i];
			auto mutated = mutate(prev);
			m_primarySamples[i] = mutated;
		}
	}

	///////////////////////////////////////////////////////

	ERPT::Path ERPT::genPath(
		scene* scene,
		sampler* sampler,
		int x, int y,
		int width, int height,
		camera* camera,
		bool willImagePlaneMutation)
	{
		// スクリーン上でのパスの変異量.
		static const int image_plane_mutation_value = 10;

#if 0
		// スクリーン上で変異する.
		auto s1 = sampler->nextSample();
		auto s2 = sampler->nextSample();

		if (willImagePlaneMutation) {
			x += int(image_plane_mutation_value * 2 * s1 - image_plane_mutation_value + 0.5);
			y += int(image_plane_mutation_value * 2 * s2 - image_plane_mutation_value + 0.5);
		}
#else
		if (willImagePlaneMutation) {
			// スクリーン上で変異する.
			auto s1 = sampler->nextSample();
			auto s2 = sampler->nextSample();

			x += int(image_plane_mutation_value * 2 * s1 - image_plane_mutation_value + 0.5);
			y += int(image_plane_mutation_value * 2 * s2 - image_plane_mutation_value + 0.5);
		}
#endif

		if (x < 0 || width <= x || y < 0 || height <= y) {
			return std::move(Path());
		}

		real u = x / (real)width;
		real v = y / (real)height;

		auto camsample = camera->sample(u, v, sampler);

		auto path = radiance(sampler, camsample.r, camera, camsample, scene);

		auto pdfOnImageSensor = camsample.pdfOnImageSensor;
		auto pdfOnLens = camsample.pdfOnLens;

		auto s = camera->getSensitivity(
			camsample.posOnImageSensor,
			camsample.posOnLens);

		Path retPath;

		retPath.contrib = path.contrib * s / (pdfOnImageSensor * pdfOnLens);
		retPath.x = x;
		retPath.y = y;
		retPath.isTerminate = path.isTerminate;

		return std::move(retPath);
	}

	void ERPT::render(
		Destination& dst,
		scene* scene,
		camera* camera)
	{
		int width = dst.width;
		int height = dst.height;
		uint32_t samples = dst.sample;
		uint32_t mutation = dst.mutation;
		vec3* color = dst.buffer;

		m_maxDepth = dst.maxDepth;
		m_rrDepth = dst.russianRouletteDepth;

		if (m_rrDepth > m_maxDepth) {
			m_rrDepth = m_maxDepth - 1;
		}

		auto threadNum = thread::getThreadNum();

		vec3 sumI(0, 0, 0);

#ifdef ENABLE_OMP
#pragma omp parallel
#endif
		{
			// edを計算.
			std::vector<vec3> tmpSumI(threadNum);

#ifdef ENABLE_OMP
#pragma omp for
#endif
			for (int y = 0; y < height; y++) {
				auto idx = thread::getThreadIdx();

				for (int x = 0; x < width; x++) {
					XorShift rnd((y * height * 4 + x * 4) * samples);
					ERPTSampler X(&rnd);

					auto path = genPath(scene, &X, x, y, width, height, camera, false);

					tmpSumI[idx] += path.contrib;
				}
			}

			for (int i = 0; i < threadNum; i++) {
				sumI += tmpSumI[i];
			}
		}

		const real ed = color::luminance(sumI / (width * height)) / mutation;

		std::vector<std::vector<vec3>> acuumImage(threadNum);
		std::vector<std::vector<vec3>> tmpImageArray(threadNum);

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int y = 0; y < height; y++) {
			AT_PRINTF("Rendering (%f)%%\n", 100.0 * y / (height - 1));

			auto idx = thread::getThreadIdx();

			auto& tmpImage = tmpImageArray[idx];
			if (tmpImage.empty()) {
				tmpImage.resize(width * height);
			}
			else {
				std::fill(tmpImage.begin(), tmpImage.end(), vec3());
			}

			for (int x = 0; x < width; x++) {
				for (uint32_t i = 0; i < samples; i++) {
					// TODO
					// sobol や halton sequence はステップ数が多すぎてオーバーフローしてしまう...
					XorShift rnd((y * height * 4 + x * 4) * samples + i + 1);
					ERPTSampler X(&rnd);

					// 現在のスクリーン上のある点からのパスによる放射輝度を求める.
					auto newSample = genPath(scene, &X, x, y, width, height, camera, false);

					// パスが光源に直接ヒットしてた場合、エネルギー分配しないで、そのまま画像に送る.
					if (newSample.isTerminate) {
						int pos = newSample.y * width + newSample.x;
						tmpImage[pos] += newSample.contrib / samples;
						continue;
					}

					const vec3 e = newSample.contrib;
					auto l = color::luminance(e);

					if (l > 0) {
						auto r = rnd.next01();
						auto illum = color::luminance(e);
						const int numChains = (int)std::floor(r + illum / (mutation * ed));;

						// 周囲に分配するエネルギー.
						const vec3 depositValue = (e / illum * ed) / samples;

						for (int nc = 0; nc < numChains; nc++) {
							ERPTSampler Y = X;
							Path Ypath = newSample;

							// Consecutive sample filtering.
							// ある点に極端にエネルギーが分配されると、スポットノイズになってしまう.
							// Unbiasedにするにはそれも仕方ないが、現実的には見苦しいのである点に対する分配回数を制限することでそのようなノイズを抑える.
							// Biasedになるが、見た目は良くなる.
							static const int MaxStack = 10;
							int stack_num = 0;
							int now_x = x;
							int now_y = y;

							for (uint32_t m = 0; m < mutation; m++) {
								ERPTSampler Z = Y;
								Z.mutate();

								Path Zpath = genPath(scene, &Z, x, y, width, height, camera, true);

								// いる？
								//Z.reset();

								auto lfz = color::luminance(Zpath.contrib);
								auto lfy = color::luminance(Ypath.contrib);

								auto q = lfz / lfy;

								auto r = rnd.next01();

								if (q > r) {
									// accept mutation.
									Y = Z;
									Ypath = Zpath;
								}

								// Consecutive sample filtering
								if (now_x == Ypath.x && now_y == Ypath.y) {
									// mutationがrejectされた回数をカウント.
									stack_num++;
								}
								else {
									// mutationがacceptされたのでreject回数をリセット.
									now_x = Ypath.x;
									now_y = Ypath.y;
									stack_num = 0;
								}

								// エネルギーをRedistributionする.
								// 同じ個所に分配され続けないように上限を制限.
								if (stack_num < MaxStack) {
#if 1
									if (!Ypath.isTerminate) {
										// 論文とは異なるが、光源に直接ヒットしたときは分配しないでみる.
										int pos = Ypath.y * width + Ypath.x;
										tmpImage[pos] += depositValue;
									}
#else
									int pos = Ypath.y * width + Ypath.x;
									tmpImage[pos] += depositValue;
#endif
								}
							}
						}
					}
				}
			}

			auto& image = acuumImage[idx];
			if (image.empty()) {
				image.resize(width * height);
			}
			for (int i = 0; i < width * height; i++) {
				image[i] += tmpImage[i];
			}
		}

		for (int n = 0; n < threadNum; n++) {
			auto& image = acuumImage[n];
			for (int i = 0; i < width * height; i++) {
				color[i] += image[i];
			}
		}
	}
}
