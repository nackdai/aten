#include "material/refraction.h"
#include "scene/hitable.h"

#pragma optimize( "", off)

namespace AT_NAME
{
	real refraction::pdf(
		const aten::MaterialParameter& param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v)
	{
		AT_ASSERT(false);

		auto ret = real(1);
		return ret;
	}

	real refraction::pdf(
		const aten::vec3& normal, 
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v) const
	{
		return pdf(m_param, normal, wi, wo, u, v);
	}

	aten::vec3 refraction::sampleDirection(
		const aten::MaterialParameter& param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		real u, real v,
		aten::sampler* sampler)
	{
		AT_ASSERT(false);

		auto reflect = wi - 2 * dot(normal, wi) * normal;
		reflect.normalize();

		return std::move(reflect);
	}

	aten::vec3 refraction::sampleDirection(
		const aten::ray& ray,
		const aten::vec3& normal,
		real u, real v,
		aten::sampler* sampler) const
	{
		return std::move(sampleDirection(m_param, normal, ray.dir, u, v, sampler));
	}

	aten::vec3 refraction::bsdf(
		const aten::MaterialParameter& param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v)
	{
		AT_ASSERT(false);

		aten::vec3 albedo = param.baseColor;
		albedo *= sampleTexture(
			(aten::texture*)param.albedoMap.ptr,
			u, v,
			real(1));

		return std::move(albedo);
	}

	aten::vec3 refraction::bsdf(
		const aten::vec3& normal, 
		const aten::vec3& wi,
		const aten::vec3& wo,
		real u, real v) const
	{
		return std::move(bsdf(m_param, normal, wi, wo, u, v));
	}

	MaterialSampling refraction::sample(
		const aten::ray& ray,
		const aten::vec3& normal,
		const aten::hitrecord& hitrec,
		aten::sampler* sampler,
		real u, real v,
		bool isLightPath/*= false*/) const
	{
		auto ret = sample(
			m_param,
			normal,
			ray.dir,
			hitrec,
			sampler,
			u, v,
			isLightPath);

		return std::move(ret);
	}

	MaterialSampling refraction::sample(
		const aten::MaterialParameter& param,
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::hitrecord& hitrec,
		aten::sampler* sampler,
		real u, real v,
		bool isLightPath/*= false*/)
	{
		MaterialSampling ret;

		// レイが入射してくる側の物体の屈折率.
		real ni = real(1);	// 真空

		// 物体内部の屈折率.
		real nt = param.ior;

		bool into = (dot(hitrec.normal, normal) > real(0));

		auto reflect = wi - 2 * dot(hitrec.normal, wi) * hitrec.normal;
		reflect.normalize();

		real cos_i = dot(wi, normal);
		real nnt = into ? ni / nt : nt / ni;

		// NOTE
		// cos_t^2 = 1 - sin_t^2
		// sin_t^2 = (ni/nt)^2 * sin_i^2 = (ni/nt)^2 * (1 - cos_i^2)
		// sin_i / sin_t = nt/ni -> sin_t = (ni/nt) * sin_i = (ni/nt) * sqrt(1 - cos_i)
		real cos_t_2 = real(1) - (nnt * nnt) * (real(1) - cos_i * cos_i);

		aten::vec3 albedo = param.baseColor;

		if (cos_t_2 < real(0)) {
			//AT_PRINTF("Reflection in refraction...\n");

			// 全反射.
			ret.pdf = real(1);
			ret.bsdf = albedo;
			ret.dir = reflect;
			ret.fresnel = real(1);

			return std::move(ret);
		}

		aten::vec3 n = into ? hitrec.normal : -hitrec.normal;
#if 0
		aten::vec3 refract = in * nnt - hitrec.normal * (into ? 1.0 : -1.0) * (cos_i * nnt + sqrt(cos_t_2));
#else
		// NOTE
		// https://www.vcl.jp/~kanazawa/raytracing/?page_id=478

		auto invnnt = 1 / nnt;
		aten::vec3 refract = nnt * (wi - (aten::sqrt(invnnt * invnnt - (1 - cos_i * cos_i)) - (-cos_i)) * normal);
#endif
		refract.normalize();

		const auto r0 = ((nt - ni) * (nt - ni)) / ((nt + ni) * (nt + ni));

		const auto c = 1 - (into ? -cos_i : dot(refract, -normal));

		// 反射方向の光が反射してray.dirの方向に運ぶ割合。同時に屈折方向の光が反射する方向に運ぶ割合.
		auto fresnel = r0 + (1 - r0) * aten::pow(c, 5);

		auto Re = fresnel;
		auto Tr = (1 - Re);

		real r = real(0.5);
		if (sampler) {
			r = sampler->nextSample();
		}

		if (param.isIdealRefraction) {
			ret.dir = refract;

			// レイの運ぶ放射輝度は屈折率の異なる物体間を移動するとき、屈折率の比の二乗の分だけ変化する.
			if (isLightPath) {
				nnt = into ? ni / nt : nt / ni;
			}
			else {
				nnt = into ? nt / ni : ni / nt;
			}

			real nnt2 = nnt * nnt;

			ret.bsdf = nnt2 * Tr * albedo;

			ret.fresnel = 0;

			ret.subpdf = real(1);
		}
		else {
			auto prob = real(0.25) + real(0.5) * Re;

			if (r < prob) {
				// 反射.
				ret.dir = reflect;
				ret.bsdf = Re * albedo;
				ret.bsdf /= prob;

				ret.subpdf = prob;

				ret.fresnel = Re;
			}
			else {
				// 屈折.
				ret.dir = refract;
				// レイの運ぶ放射輝度は屈折率の異なる物体間を移動するとき、屈折率の比の二乗の分だけ変化する.
				if (isLightPath) {
					nnt = into ? ni / nt : nt / ni;
				}
				else {
					nnt = into ? nt / ni : ni / nt;
				}

				real nnt2 = nnt * nnt;

				ret.bsdf = nnt2 * Tr * albedo;
				ret.bsdf /= (1 - prob);

				ret.subpdf = (1 - prob);

				ret.fresnel = Tr;
			}
		}

		ret.pdf = 1;

		return std::move(ret);
	}

	refraction::RefractionSampling refraction::check(
		material* mtrl,
		const aten::vec3& in,
		const aten::vec3& normal,
		const aten::vec3& orienting_normal)
	{
		if (!mtrl->isSingular() || !mtrl->isTranslucent()) {
			return std::move(RefractionSampling(false, real(0), real(0)));
		}

		// レイが入射してくる側の物体の屈折率.
		real ni = real(1);	// 真空

		// 物体内部の屈折率.
		real nt = mtrl->ior();

		bool into = (dot(normal, orienting_normal) > real(0));

		auto reflect = in - 2 * dot(normal, in) * normal;
		reflect.normalize();

		real cos_i = dot(in, normal);
		real nnt = into ? ni / nt : nt / ni;

		// NOTE
		// cos_t^2 = 1 - sin_t^2
		// sin_t^2 = (ni/nt)^2 * sin_i^2 = (ni/nt)^2 * (1 - cos_i^2)
		// sin_i / sin_t = nt/ni -> sin_t = (ni/nt) * sin_i = (ni/nt) * sqrt(1 - cos_i)
		real cos_t_2 = real(1) - (nnt * nnt) * (real(1) - cos_i * cos_i);

		aten::vec3 albedo = mtrl->color();

		if (cos_t_2 < real(0)) {
			return std::move(RefractionSampling(false, real(1), real(0)));
		}

		aten::vec3 n = into ? normal : -normal;
#if 0
		aten::vec3 refract = in * nnt - hitrec.normal * (into ? 1.0 : -1.0) * (cos_i * nnt + sqrt(cos_t_2));
#else
		// NOTE
		// https://www.vcl.jp/~kanazawa/raytracing/?page_id=478

		auto invnnt = 1 / nnt;
		aten::vec3 refract = nnt * (in - (aten::sqrt(invnnt * invnnt - (1 - cos_i * cos_i)) - (-cos_i)) * normal);
#endif
		refract.normalize();

		const auto r0 = ((nt - ni) * (nt - ni)) / ((nt + ni) * (nt + ni));

		const auto c = 1 - (into ? -cos_i : dot(refract, -normal));

		// 反射方向の光が反射してray.dirの方向に運ぶ割合。同時に屈折方向の光が反射する方向に運ぶ割合.
		auto fresnel = r0 + (1 - r0) * aten::pow(c, 5);

		// レイの運ぶ放射輝度は屈折率の異なる物体間を移動するとき、屈折率の比の二乗の分だけ変化する.
		real nn = nnt * nnt;

		auto Re = fresnel;
		auto Tr = (1 - Re) * nn;

		refraction* refr = static_cast<refraction*>(mtrl);

		if (refr->isIdealRefraction()) {
			return std::move(RefractionSampling(true, real(0), real(1), true));
		}
		else {
			auto prob = 0.25 + 0.5 * Re;
			return std::move(RefractionSampling(true, real(prob), real(1 - prob)));
		}
	}
}
