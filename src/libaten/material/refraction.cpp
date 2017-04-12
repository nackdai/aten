#include "material/refraction.h"
#include "scene/hitable.h"

#pragma optimize( "", off)

namespace aten
{
	real refraction::pdf(
		const vec3& normal, 
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		AT_ASSERT(false);

		auto ret = real(1);
		return ret;
	}

	vec3 refraction::sampleDirection(
		const ray& ray,
		const vec3& normal,
		real u, real v,
		sampler* sampler) const
	{
		AT_ASSERT(false);

		const vec3& in = ray.dir;

		auto reflect = in - 2 * dot(normal, in) * normal;
		reflect.normalize();

		return std::move(reflect);
	}

	vec3 refraction::bsdf(
		const vec3& normal, 
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		AT_ASSERT(false);

		vec3 bsdf = color();
		bsdf *= sampleAlbedoMap(u, v);

		return std::move(bsdf);
	}

	material::sampling refraction::sample(
		const ray& ray,
		const vec3& normal,
		const hitrecord& hitrec,
		sampler* sampler,
		real u, real v) const
	{
		sampling ret;

		const vec3& in = ray.dir;

		// ƒŒƒC‚ª“üŽË‚µ‚Ä‚­‚é‘¤‚Ì•¨‘Ì‚Ì‹üÜ—¦.
		real ni = real(1);	// ^‹ó

		// •¨‘Ì“à•”‚Ì‹üÜ—¦.
		real nt = ior();

		bool into = (dot(hitrec.normal, normal) > real(0));

		auto reflect = in - 2 * dot(hitrec.normal, in) * hitrec.normal;
		reflect.normalize();

		real cos_i = dot(in, normal);
		real nnt = into ? ni / nt : nt / ni;

		// NOTE
		// cos_t^2 = 1 - sin_t^2
		// sin_t^2 = (ni/nt)^2 * sin_i^2 = (ni/nt)^2 * (1 - cos_i^2)
		// sin_i / sin_t = nt/ni -> sin_t = (ni/nt) * sin_i = (ni/nt) * sqrt(1 - cos_i)
		real cos_t_2 = real(1) - (nnt * nnt) * (real(1) - cos_i * cos_i);

		vec3 albedo = color();

		if (cos_t_2 < real(0)) {
			//AT_PRINTF("Reflection in refraction...\n");

			// ‘S”½ŽË.
			ret.pdf = real(1);
			ret.bsdf = albedo;
			ret.dir = reflect;

#if 0
			// For canceling cosine factor.
			auto c = dot(normal, ret.dir);
			if (c > real(0)) {
				ret.bsdf = albedo / c;
			}
#endif

			return std::move(ret);
		}

		vec3 n = into ? hitrec.normal : -hitrec.normal;
#if 0
		vec3 refract = in * nnt - hitrec.normal * (into ? 1.0 : -1.0) * (cos_i * nnt + sqrt(cos_t_2));
#else
		// NOTE
		// https://www.vcl.jp/~kanazawa/raytracing/?page_id=478

		auto invnnt = 1 / nnt;
		vec3 refract = nnt * (in - (aten::sqrt(invnnt * invnnt - (1 - cos_i * cos_i)) - (-cos_i)) * normal);
#endif
		refract.normalize();

		const auto r0 = ((nt - ni) * (nt - ni)) / ((nt + ni) * (nt + ni));

		const auto c = 1 - (into ? -cos_i : dot(refract, -normal));

		// ”½ŽË•ûŒü‚ÌŒõ‚ª”½ŽË‚µ‚Äray.dir‚Ì•ûŒü‚É‰^‚ÔŠ„‡B“¯Žž‚É‹üÜ•ûŒü‚ÌŒõ‚ª”½ŽË‚·‚é•ûŒü‚É‰^‚ÔŠ„‡.
		auto fresnel = r0 + (1 - r0) * aten::pow(c, 5);

		// ƒŒƒC‚Ì‰^‚Ô•úŽË‹P“x‚Í‹üÜ—¦‚ÌˆÙ‚È‚é•¨‘ÌŠÔ‚ðˆÚ“®‚·‚é‚Æ‚«A‹üÜ—¦‚Ì”ä‚Ì“ñæ‚Ì•ª‚¾‚¯•Ï‰»‚·‚é.
		real nn = nnt * nnt;

		auto Re = fresnel;
		auto Tr = (1 - Re) * nn;

		real r = real(0.5);
		if (sampler) {
			r = sampler->nextSample();
		}

		if (isIdealRefraction()) {
			ret.dir = refract;
			ret.bsdf = Tr * albedo;
			ret.fresnel = 0;

			ret.subpdf = real(1);

#if 0
			// For canceling cosine factor.
			auto denom = dot(normal, refract);
			ret.bsdf /= denom;
#endif
		}
		else {
			auto prob = real(0.25) + real(0.5) * Re;
#if 1
			if (r < prob) {
				// ”½ŽË.
				ret.dir = reflect;
				ret.bsdf = Re * albedo;
				ret.bsdf /= prob;

				ret.subpdf = prob;

				ret.fresnel = Re;
			}
			else {
				// ‹üÜ.
				ret.dir = refract;
				ret.bsdf = Tr * albedo;
				ret.bsdf /= (real(1) - prob);

				ret.subpdf = (real(1) - prob);

				ret.fresnel = 0;
			}
#else
			if (r < prob) {
				// ”½ŽË.
				ret.dir = reflect;

				// For canceling cosine factor.
				auto denom = dot(normal, reflect);
				ret.bsdf = Re * albedo / denom;
				ret.bsdf /= prob;

				ret.subpdf = prob;
			}
			else {
				// ‹üÜ.
				ret.dir = refract;

				// For canceling cosine factor.
				auto denom = dot(normal, refract);
				ret.bsdf = Tr * albedo / denom;
				ret.bsdf /= (1 - prob);

				ret.subpdf = (1 - prob);
			}
#endif
		}

		ret.pdf = 1;

		return std::move(ret);
	}

	refraction::RefractionSampling refraction::check(
		material* mtrl,
		const vec3& in,
		const vec3& normal,
		const vec3& orienting_normal)
	{
		if (!mtrl->isSingular() || !mtrl->isTranslucent()) {
			return std::move(RefractionSampling(false, real(0), real(0)));
		}

		// ƒŒƒC‚ª“üŽË‚µ‚Ä‚­‚é‘¤‚Ì•¨‘Ì‚Ì‹üÜ—¦.
		real ni = real(1);	// ^‹ó

		// •¨‘Ì“à•”‚Ì‹üÜ—¦.
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

		vec3 albedo = mtrl->color();

		if (cos_t_2 < real(0)) {
			return std::move(RefractionSampling(false, real(1), real(0)));
		}

		vec3 n = into ? normal : -normal;
#if 0
		vec3 refract = in * nnt - hitrec.normal * (into ? 1.0 : -1.0) * (cos_i * nnt + sqrt(cos_t_2));
#else
		// NOTE
		// https://www.vcl.jp/~kanazawa/raytracing/?page_id=478

		auto invnnt = 1 / nnt;
		vec3 refract = nnt * (in - (aten::sqrt(invnnt * invnnt - (1 - cos_i * cos_i)) - (-cos_i)) * normal);
#endif
		refract.normalize();

		const auto r0 = ((nt - ni) * (nt - ni)) / ((nt + ni) * (nt + ni));

		const auto c = 1 - (into ? -cos_i : dot(refract, -normal));

		// ”½ŽË•ûŒü‚ÌŒõ‚ª”½ŽË‚µ‚Äray.dir‚Ì•ûŒü‚É‰^‚ÔŠ„‡B“¯Žž‚É‹üÜ•ûŒü‚ÌŒõ‚ª”½ŽË‚·‚é•ûŒü‚É‰^‚ÔŠ„‡.
		auto fresnel = r0 + (1 - r0) * aten::pow(c, 5);

		// ƒŒƒC‚Ì‰^‚Ô•úŽË‹P“x‚Í‹üÜ—¦‚ÌˆÙ‚È‚é•¨‘ÌŠÔ‚ðˆÚ“®‚·‚é‚Æ‚«A‹üÜ—¦‚Ì”ä‚Ì“ñæ‚Ì•ª‚¾‚¯•Ï‰»‚·‚é.
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

	void refraction::serialize(MaterialParam& param) const
	{
		material::serialize(this, param);

		// TODO
#if 0
		param.isIdealRefraction = m_isIdealRefraction;
#endif

		// TODO
	}
}
