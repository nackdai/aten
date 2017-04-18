#include <atomic>
#include "material/material.h"
#include "light/light.h"

namespace aten
{
	// NOTE
	// 0 は予約済みなので、1 から始める.
	static std::atomic<uint32_t> g_id = 1;

	material::material()
		: m_param(MaterialType())
	{
		m_id = g_id.fetch_add(1);
	}

	material::material(
		const MaterialType& type,
		const vec3& clr,
		real ior/*= 1*/,
		texture* albedoMap/*= nullptr*/,
		texture* normalMap/*= nullptr*/)
		: m_param(type)
	{
		m_id = g_id.fetch_add(1);

		m_param.baseColor = clr;
		m_param.ior = ior;
		m_param.albedoMap.tex = albedoMap;
		m_param.normalMap.tex = normalMap;
	}

	material::material(const MaterialType& type, Values& val)
		: m_param(type)
	{
		m_id = g_id.fetch_add(1);

		m_param.baseColor = val.get("color", m_param.baseColor);
		m_param.ior = val.get("ior", m_param.ior);
		m_param.albedoMap.tex = (texture*)val.get("albedomap", (void*)m_param.albedoMap.tex);
		m_param.normalMap.tex = (texture*)val.get("normalmap", (void*)m_param.normalMap.tex);
	}

	NPRMaterial::NPRMaterial(const vec3& e, Light* light)
		: material(MaterialTypeNPR, e)
	{
		setTargetLight(light);
	}

	void NPRMaterial::setTargetLight(Light* light)
	{
		m_targetLight = light;
	}

	const Light* NPRMaterial::getTargetLight() const
	{
		return m_targetLight;
	}

	// NOTE
	// Schlick によるフレネル反射率の近似.
	// http://yokotakenji.me/log/math/4501/
	// https://en.wikipedia.org/wiki/Schlick%27s_approximation

	// NOTE
	// フレネル反射率について.
	// http://d.hatena.ne.jp/hanecci/20130525/p3

	real schlick(
		const vec3& in,
		const vec3& normal,
		real ni, real nt)
	{
		// NOTE
		// Fschlick(v,h) ≒ R0 + (1 - R0)(1 - cosΘ)^5
		// R0 = ((n1 - n2) / (n1 + n2))^2

		auto r0 = (ni - nt) / (ni + nt);
		r0 = r0 * r0;

		auto c = dot(in, normal);

		return r0 + (1 - r0) * aten::pow((1 - c), 5);
	}

	real computFresnel(
		const vec3& in,
		const vec3& normal,
		real ni, real nt)
	{
		real cos_i = dot(in, normal);

		bool isEnter = (cos_i > real(0));

		vec3 n = normal;

		if (isEnter) {
			// レイが出ていくので、全部反対.
			auto tmp = nt;
			nt = real(1);
			ni = tmp;

			n = -n;
		}

		auto eta = ni / nt;

		auto sini2 = 1.f - cos_i * cos_i;
		auto sint2 = eta * eta * sini2;

		auto fresnel = schlick(
			in, 
			n, ni, nt);

		return fresnel;
	}

	void material::applyNormalMap(
		const vec3& orgNml,
		vec3& newNml,
		real u, real v) const
	{
		if (m_param.normalMap.tex) {
			newNml = ((texture*)m_param.normalMap.tex)->at(u, v);
			newNml = 2 * newNml - vec3(1);
			newNml.normalize();

			vec3 n = normalize(orgNml);
			vec3 t = getOrthoVector(n);
			vec3 b = cross(n, t);

			newNml = newNml.z * n + newNml.x * t + newNml.y * b;
			newNml.normalize();
		}
		else {
			newNml = normalize(orgNml);
		}
	}

	real material::computeFresnel(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real outsideIor/*= 1*/) const
	{
		vec3 V = -wi;
		vec3 L = wo;
		vec3 N = normal;
		vec3 H = normalize(L + V);

		auto ni = outsideIor;
		auto nt = ior();

		// NOTE
		// Fschlick(v,h) ≒ R0 + (1 - R0)(1 - cosΘ)^5
		// R0 = ((n1 - n2) / (n1 + n2))^2

		auto r0 = (ni - nt) / (ni + nt);
		r0 = r0 * r0;

		auto LdotH = aten::abs(dot(L, H));

		auto F = r0 + (1 - r0) * aten::pow((1 - LdotH), 5);

		return F;
	}
}
