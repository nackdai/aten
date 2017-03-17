#include "material/layer.h"
#include "scene/hitable.h"

namespace aten
{
	vec3 LayeredBSDF::sampleAlbedoMap(real u, real v) const
	{
		auto num = m_layer.size();

		vec3 albedo(1);

		for (int i = 0; i < num; i++) {
			auto mtrl = m_layer[i];

			// TODO
			auto c = mtrl->color();
			auto a = mtrl->sampleAlbedoMap(u, v);

			albedo *= c * a;
		}

		return std::move(albedo);
	}

	void LayeredBSDF::applyNormalMap(
		const vec3& orgNml,
		vec3& newNml,
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
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real outsideIor/*= 1*/) const
	{
		// TODO
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

	void LayeredBSDF::add(material* mtrl)
	{
		m_layer.push_back(mtrl);
	}

	material::sampling LayeredBSDF::sample(
		const ray& ray,
		const vec3& normal,
		const hitrecord& hitrec,
		sampler* sampler,
		real u, real v) const
	{
		sampling ret;

		auto num = m_layer.size();

		if (num == 0) {
			AT_ASSERT(false);
			return std::move(ret);
		}

		real weight = 1;

		for (int i = 0; i < num; i++) {
			auto mtrl = m_layer[i];

			vec3 appliedNml = normal;

			// NOTE
			// 外部では最表層の NormalMap が適用されている.
			if (i > 0) {
				mtrl->applyNormalMap(normal, appliedNml, u, v);
			}

			auto sampleres = mtrl->sample(ray, appliedNml, hitrec, sampler, u, v);

			const auto f = aten::clamp<real>(sampleres.fresnel, 0, 1);

			ret.pdf += weight * f * sampleres.pdf;
			ret.bsdf += weight * f * sampleres.bsdf;

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
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v,
		sampler* sampler) const
	{
		auto num = m_layer.size();
		
		real pdf = 0;

		real weight = 1;
		real ior = 1;	// 真空から始める.

		for (int i = 0; i < num; i++) {
			auto mtrl = m_layer[i];

			vec3 appliedNml = normal;

			// NOTE
			// 外部では最表層の NormalMap が適用されている.
			if (i > 0) {
				mtrl->applyNormalMap(normal, appliedNml, u, v);
			}

			auto p = mtrl->pdf(appliedNml, wi, wo, u, v, sampler);
			auto f = mtrl->computeFresnel(appliedNml, wi, wo, ior);

			f = aten::clamp<real>(f, 0, 1);

			pdf += weight * f * p;

			weight = aten::clamp<real>(weight - f, 0, 1);
			if (weight <= 0) {
				break;
			}

			// 上層の値を下層に使う.
			ior = mtrl->ior();
		}

		return pdf;
	}

	vec3 LayeredBSDF::sampleDirection(
		const ray& ray,
		const vec3& normal,
		real u, real v,
		sampler* sampler) const
	{
		auto num = m_layer.size();
		AT_ASSERT(num > 0);

		auto mtrl = m_layer[0];
		
		auto dir = mtrl->sampleDirection(ray, normal, u, v, sampler);

		return std::move(dir);
	}

	vec3 LayeredBSDF::bsdf(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		auto num = m_layer.size();

		vec3 bsdf;

		real weight = 1;
		real ior = 1;	// 真空から始める.

		for (int i = 0; i < num; i++) {
			auto mtrl = m_layer[i];

			vec3 appliedNml = normal;

			// NOTE
			// 外部では最表層の NormalMap が適用されている.
			if (i > 0) {
				mtrl->applyNormalMap(normal, appliedNml, u, v);
			}

			auto b = mtrl->bsdf(appliedNml, wi, wo, u, v);
			auto f = mtrl->computeFresnel(appliedNml, wi, wo, ior);

			f = aten::clamp<real>(f, 0, 1);

			bsdf += weight * f * b;

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
