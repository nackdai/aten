#include <atomic>
#include "material/material.h"
#include "light/light.h"

namespace AT_NAME
{
	std::vector<material*> material::g_materials;

	uint32_t material::getMaterialNum()
	{
		return (uint32_t)g_materials.size();
	}

	material* material::getMaterial(uint32_t idx)
	{
		if (idx < g_materials.size()) {
			return g_materials[idx];
		}
		return nullptr;
	}

	int material::findMaterialIdx(material* mtrl)
	{
		auto found = std::find(g_materials.begin(), g_materials.end(), mtrl);
		if (found != g_materials.end()) {
			auto id = std::distance(g_materials.begin(), found);
			AT_ASSERT(mtrl == g_materials[id]);
			return id;
		}
		return -1;
	}

	const std::vector<material*>& material::getMaterials()
	{
		return g_materials;
	}

	material::material(aten::MaterialType type, const aten::MaterialAttribute& attrib)
		: m_param(type, attrib)
	{
		m_id = g_materials.size();
		g_materials.push_back(this);
	}

	material::material(
		aten::MaterialType type,
		const aten::MaterialAttribute& attrib,
		const aten::vec3& clr,
		real ior/*= 1*/,
		aten::texture* albedoMap/*= nullptr*/,
		aten::texture* normalMap/*= nullptr*/)
		: m_param(type, attrib)
	{
		m_id = g_materials.size();
		g_materials.push_back(this);

		m_param.baseColor = clr;
		m_param.ior = ior;
		m_param.albedoMap.ptr = albedoMap;
		m_param.normalMap.ptr = normalMap;
	}

	material::material(aten::MaterialType type, const aten::MaterialAttribute& attrib, aten::Values& val)
		: m_param(type, attrib)
	{
		m_id = g_materials.size();
		g_materials.push_back(this);

		m_param.baseColor = val.get("color", m_param.baseColor);
		m_param.ior = val.get("ior", m_param.ior);
		m_param.albedoMap.ptr = (aten::texture*)val.get("albedomap", (void*)m_param.albedoMap.ptr);
		m_param.normalMap.ptr = (aten::texture*)val.get("normalmap", (void*)m_param.normalMap.ptr);
	}

	material::~material()
	{
		auto found = std::find(g_materials.begin(), g_materials.end(), this);
		if (found != g_materials.end()) {
			g_materials.erase(found);
		}
	}

	NPRMaterial::NPRMaterial(
		aten::MaterialType type,
		const aten::vec3& e, AT_NAME::Light* light)
		: material(type, MaterialAttributeNPR, e)
	{
		setTargetLight(light);
	}

	void NPRMaterial::setTargetLight(AT_NAME::Light* light)
	{
		m_targetLight = light;
	}

	const AT_NAME::Light* NPRMaterial::getTargetLight() const
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
		const aten::vec3& in,
		const aten::vec3& normal,
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
		const aten::vec3& in,
		const aten::vec3& normal,
		real ni, real nt)
	{
		real cos_i = dot(in, normal);

		bool isEnter = (cos_i > real(0));

		aten::vec3 n = normal;

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
		const aten::vec3& orgNml,
		aten::vec3& newNml,
		real u, real v) const
	{
		if (m_param.normalMap.ptr) {
			newNml = ((aten::texture*)m_param.normalMap.ptr)->at(u, v);
			newNml = real(2) * newNml - aten::vec3(1);
			newNml = normalize(newNml);

			aten::vec3 n = normalize(orgNml);
			aten::vec3 t = aten::getOrthoVector(n);
			aten::vec3 b = cross(n, t);

			newNml = newNml.z * n + newNml.x * t + newNml.y * b;
			newNml = normalize(newNml);
		}
		else {
			newNml = normalize(orgNml);
		}
	}

	real material::computeFresnel(
		const aten::vec3& normal,
		const aten::vec3& wi,
		const aten::vec3& wo,
		real outsideIor/*= 1*/) const
	{
		aten::vec3 V = -wi;
		aten::vec3 L = wo;
		aten::vec3 N = normal;
		aten::vec3 H = normalize(L + V);

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
