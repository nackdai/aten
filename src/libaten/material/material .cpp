#include "material/material.h"

namespace aten
{
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

		float sini2 = 1.f - cos_i * cos_i;
		float sint2 = eta * eta * sini2;

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
		if (m_normalMap) {
			newNml = m_normalMap->at(u, v);
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
