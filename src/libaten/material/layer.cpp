#include "material/layer.h"
#include "scene/hitable.h"

namespace AT_NAME
{
	aten::vec3 LayeredBSDF::sampleAlbedoMap(real u, real v) const
	{
		auto num = m_layer.size();

		aten::vec3 albedo = aten::vec3(1);

		for (int i = 0; i < num; i++) {
			auto mtrl = m_layer[i];

			// TODO
			auto c = mtrl->color();
			auto a = mtrl->sampleAlbedoMap(u, v);

			albedo *= c * a;
		}

		return std::move(albedo);
	}

	bool LayeredBSDF::isGlossy() const
	{
		auto num = m_layer.size();

		for (int i = 0; i < num; i++) {
			auto mtrl = m_layer[i];

			if (mtrl->isGlossy()) {
				return true;
			}
		}

		return false;
	}

	void LayeredBSDF::applyNormalMap(
		const aten::vec3& orgNml,
		aten::vec3& newNml,
		real u, real v) const
	{
		auto num = m_layer.size();

		if (num == 0) {
			newNml = orgNml;
		}
		else {
			// 最表層の NormalMap を適用.
			auto mtrl = m_layer[0];
			mtrl->applyNormalMap(orgNml, newNml, u, v);
		}
	}

	real LayeredBSDF::computeFresnel(
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real outsideIor/*= 1*/) const
	{
		// TODO
		// Not permit layer in layer, so this api should not be called.
		AT_ASSERT(false);

		auto num = m_layer.size();

		if (num == 0) {
			return real(1);
		}
		else {
			// 最表層のフレネルを返す.
			auto mtrl = m_layer[0];
			auto f = mtrl->computeFresnel(normal, wi, wo, outsideIor);
			return f;
		}
	}

	bool LayeredBSDF::add(material* mtrl)
	{
		// TODO
		// GPU側と処理を合わせるため、３層までにする.
		if (m_layer.size() > 3) {
			AT_ASSERT(false);
			return false;
		}

		// Not permit layer in layer.
		if (mtrl->param().type == aten::MaterialType::Layer) {
			AT_ASSERT(false);
			return false;
		}

		m_param.layer[m_layer.size()] = mtrl->id();
		m_layer.push_back(mtrl);

		return true;
	}

	MaterialSampling LayeredBSDF::sample(
		const aten::ray& ray,
		const aten::vec3& normal,
		const aten::vec3& orgnormal,
		aten::sampler* sampler,
		real u, real v,
		bool isLightPath/*= false*/) const
	{
		MaterialSampling ret;

		auto num = m_layer.size();

		if (num == 0) {
			AT_ASSERT(false);
			return std::move(ret);
		}

		real weight = 1;

		for (int i = 0; i < num; i++) {
			auto mtrl = m_layer[i];

			aten::vec3 appliedNml = normal;

			// NOTE
			// 外部では最表層の NormalMap が適用されているので、下層レイヤーのマテリアルごとに法線マップを適用する.
			if (i > 0) {
				mtrl->applyNormalMap(normal, appliedNml, u, v);
			}

			auto sampleres = mtrl->sample(ray, appliedNml, orgnormal, sampler, u, v);

			const auto f = aten::clamp<real>(sampleres.fresnel, 0, 1);

			ret.pdf += weight * f * sampleres.pdf;

			// bsdf includes fresnale value.
			ret.bsdf += weight * sampleres.bsdf;
			//ret.bsdf += weight * f * sampleres.bsdf;

			// TODO
			// ret.fresnel

			weight = aten::clamp<real>(weight - f, 0, 1);
			if (weight <= 0) {
				break;
			}

			if (i == 0) {
				ret.dir = sampleres.dir;
			}
		}

		return std::move(ret);
	}

	real LayeredBSDF::pdf(
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v) const
	{
		auto num = m_layer.size();
		
		real pdf = 0;

		real weight = 1;
		real ior = 1;	// 真空から始める.

		for (int i = 0; i < num; i++) {
			auto mtrl = m_layer[i];

			aten::vec3 appliedNml = normal;

			// NOTE
			// 外部では最表層の NormalMap が適用されているので、下層レイヤーのマテリアルごとに法線マップを適用する.
			if (i > 0) {
				mtrl->applyNormalMap(normal, appliedNml, u, v);
			}

			auto p = mtrl->pdf(appliedNml, wi, wo, u, v);
			auto f = mtrl->computeFresnel(appliedNml, wi, wo, ior);

			f = aten::clamp<real>(f, 0, 1);

			pdf += weight * p;

			weight = aten::clamp<real>(weight - f, 0, 1);
			if (weight <= 0) {
				break;
			}

			// 上層の値を下層に使う.
			ior = mtrl->ior();
		}

		return pdf;
	}

	aten::vec3 LayeredBSDF::sampleDirection(
		const aten::ray& ray,
		const aten::vec3& normal,
		real u, real v,
		aten::sampler* sampler) const
	{
		auto num = m_layer.size();
		AT_ASSERT(num > 0);

		auto mtrl = m_layer[0];
		
		auto dir = mtrl->sampleDirection(ray, normal, u, v, sampler);

		return std::move(dir);
	}

	aten::vec3 LayeredBSDF::bsdf(
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v) const
	{
		auto num = m_layer.size();

		aten::vec3 bsdf;

		real weight = 1;
		real ior = 1;	// 真空から始める.

		for (int i = 0; i < num; i++) {
			auto mtrl = m_layer[i];

			aten::vec3 appliedNml = normal;

			// NOTE
			// 外部では最表層の NormalMap が適用されているので、下層レイヤーのマテリアルごとに法線マップを適用する.
			if (i > 0) {
				mtrl->applyNormalMap(normal, appliedNml, u, v);
			}

			auto b = mtrl->bsdf(appliedNml, wi, wo, u, v);
			auto f = mtrl->computeFresnel(appliedNml, wi, wo, ior);

			f = aten::clamp<real>(f, 0, 1);

			// bsdf includes fresnel value.
			bsdf += weight * b;
			//bsdf += weight * f * b;

			weight = aten::clamp<real>(weight - f, 0, 1);
			if (weight <= 0) {
				break;
			}

			// 上層の値を下層に使う.
			ior = mtrl->ior();
		}

		return std::move(bsdf);
	}
}
